import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

from .attn import MultiHeadAttention
from .utils import unpad_padded
from .xfmr import Residual, PreNorm, PositionwiseFeedForward, key_padding_mask


class MC_PreNorm(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.model = model

    def forward(self, x, add):
        return self.model(self.norm(x), self.norm(add))


class MC_Residual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, *inputs):
        initial = inputs[0]
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs + initial


class MC_Applier(nn.Module):
    def __init__(self, model, applier):
        super().__init__()
        self.model = model
        self.applier = applier

    def forward(self, x, lip_add):
        return self.applier(self.model, x, lip_add)


class MC_TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, rpe_k=0):
        super().__init__()
        attn = MultiHeadAttention(dim, heads, dropout, rpe_k)
        ffn = PositionwiseFeedForward(dim, 4 * dim, dropout)
        MC_wrap = lambda m: MC_Residual(MC_PreNorm(dim, m), nn.Dropout(dropout))
        wrap = lambda m: Residual(PreNorm(dim, m), nn.Dropout(dropout))
        self.attn = MC_wrap(MC_Applier(attn, lambda m, x, add: m(x, add, add, self.xm)[0]))
        self.ffn = wrap(ffn)

    def forward(self, x, add, xm):
        # hack the mask here
        self.xm = xm
        x = self.attn(x, add)
        del self.xm
        x = self.ffn(x)
        return x


class MC_TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, num_layers, dropout=0.1, rpe_k=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for i in range(num_layers):
            self.layers += [
                MC_TransformerEncoderLayer(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rpe_k=rpe_k,
                )
            ]

    def forward(self, x, add):
        """
        Args:
            x: [(t d)]
        Returns:
            x: [(t d)]
        """
        xl = list(map(len, x))
        x = pad_sequence(x, True)
        add = pad_sequence(add, True)
        xm = key_padding_mask(xl).to(x.device)
        xm = xm.unsqueeze(dim=1)  # repeat mask for all targets
        for layer in self.layers:
            x = layer(x, add, xm)
        x = self.norm(x)
        return unpad_padded(x, xl)