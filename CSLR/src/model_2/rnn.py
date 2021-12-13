import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .utils import unpad_padded

class RNN(nn.Module):
    def __init__(self, dim=512, num_layers=2, dropout=0.1):
        super(RNN, self).__init__()
        self.hidden_size = dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=dim, hidden_size=dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)

        return h0

    def forward(self, x):
        xl = list(map(len, x))
        x = pad_sequence(x, True)
        hidden = self.init_hidden(x.shape[0])
        x = torch.nn.utils.rnn.pack_padded_sequence(x, xl, enforce_sorted=False, batch_first=True)
        x = self.rnn(x, hidden)[0]
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.data
        x = self.dropout(self.norm(x))
        return unpad_padded(x, xl)