import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
import numpy as np

from .attn import MultiHeadAttention
from .utils import unpad_padded


def key_padding_mask(l):
    """Blank is True
    Args:
        l: lenghts (b)
    Returns:
        mask: (b l)
    """
    mask = torch.zeros(len(l), max(l)).bool()
    for i, li in enumerate(l):
        mask[i, li:] = True
    return mask


class PreNorm(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))

class MultiChannelPreNorm(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.model = model

    def forward(self, x1, x2):
        return self.model(self.norm1(x1), self.norm2(x2))

class Residual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, x):
        return super().forward(x) + x

class MultiChannelResidual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, *inputs):
        x1 = inputs[0]
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs + x1

class Applier(nn.Module):
    def __init__(self, model, applier):
        super().__init__()
        self.model = model
        self.applier = applier

    def forward(self, x):
        return self.applier(self.model, x)

class MultiChannelApplier(nn.Module):
    def __init__(self, model, applier):
        super().__init__()
        self.model = model
        self.applier = applier

    def forward(self, x1, x2):
        return self.applier(self.model, x1, x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, rpe_k=0):
        super().__init__()
        attn = MultiHeadAttention(dim, heads, dropout, rpe_k)
        ffn = PositionwiseFeedForward(dim, 4 * dim, dropout)
        wrap = lambda m: Residual(PreNorm(dim, m), nn.Dropout(dropout))
        self.attn = wrap(Applier(attn, lambda m, x: m(x, x, x, self.xm)[0]))
        self.ffn = wrap(ffn)

    def forward(self, x, xm):
        # hack the mask here
        self.xm = xm
        x = self.attn(x)
        del self.xm
        x = self.ffn(x)
        return x

class MultiChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, rpe_k=0):
        super().__init__()
        attn = MultiHeadAttention(dim, heads, dropout, rpe_k)
        ffn = PositionwiseFeedForward(dim, 4 * dim, dropout)
        wrap = lambda m: Residual(PreNorm(dim, m), nn.Dropout(dropout))
        MultiChannelwrap = lambda m: MultiChannelResidual(MultiChannelPreNorm(dim, m), nn.Dropout(dropout))
        self.attn = MultiChannelwrap(MultiChannelApplier(attn, lambda m, x1, x2: m(x1, x2, x2, self.xm)[0]))
        self.ffn = wrap(ffn)

    def forward(self, x1, x2, xm):
        # hack the mask here
        self.xm = xm
        x = self.attn(x1, x2)
        del self.xm
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, num_layers, dropout=0.1, rpe_k=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for i in range(num_layers):
            self.layers += [
                TransformerEncoderLayer(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rpe_k=rpe_k,
                )
            ]

    def forward(self, x):
        """
        Args:
            x: [(t d)]
        Returns:
            x: [(t d)]
        """
        xl = list(map(len, x))
        x = pad_sequence(x, True)
        xm = key_padding_mask(xl).to(x.device)
        xm = xm.unsqueeze(dim=1)  # repeat mask for all targets
        for layer in self.layers:
            x = layer(x, xm)
        x = self.norm(x)
        return unpad_padded(x, xl)

class MultiChannelTransformerEncoder(nn.Module):
    def __init__(self, dim, heads, num_layers, dropout=0.1, rpe_k=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for i in range(num_layers):
            self.layers += [
                MultiChannelTransformerEncoderLayer(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rpe_k=rpe_k,
                )
            ]

    def forward(self, x1, x2):
        """
        Args:
            x: [(t d)]
        Returns:
            x: [(t d)]
        """
        xl = list(map(len, x1))
        x1 = pad_sequence(x1, True)
        x2 = pad_sequence(x2, True)
        xm = key_padding_mask(xl).to(x1.device)
        xm = xm.unsqueeze(dim=1)  # repeat mask for all targets
        for layer in self.layers:
            x = layer(x1, x2, xm)
        x = self.norm(x)
        return unpad_padded(x, xl)