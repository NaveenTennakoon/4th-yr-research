import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from .sgs import SGSResNet18
from .xfmr import TransformerEncoder
from .dec import Decoder


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_num_states,
        dim=512,
        rpe_k=8,
        heads=4,
        semantic_layers=2,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size: vocabulary size of the dataset.
            max_num_states: max number of state per gloss.
            dim: hidden dim for transformer encoder.
            p_detach: gradient stopping proportion.
            rpe_k: the window size (one side) for relative postional encoding.
            heads: number of heads for transformer encoder.
            semantic_layers: number of layers for transformer encoder.
            dropout: p_dropout.
        """
        super().__init__()
        self.max_num_states = max_num_states
        self.vocab_size = vocab_size

        self.visual = SGSResNet18(
            dim,
        )

        self.semantic = TransformerEncoder(
            dim,
            heads,
            semantic_layers,
            dropout,
            rpe_k,
        )

        self.decoder = Decoder(
            vocab_size,
            max_num_states,
        )

        # plus 1 for blank
        self.classifier = nn.Linear(dim, self.decoder.total_states + 1, bias=False)

        self.blank = self.decoder.total_states  # last dim as blank

    def forward(self, x):
        """
        Args:
            x: list of (t c h w)
        Return:
            log probs [(t n)]
        """
        xl = list(map(len, x))
        x = self.visual(x)
        x = self.semantic(x)
        x = torch.cat(x)
        x = self.classifier(x)
        x = x.log_softmax(dim=-1)
        x = x.split(xl)
        return x

    def expand(self, y, n=None):
        """Expand to tensor"""
        return torch.tensor(self.decoder.expand(y, n)).to(y.device)

    def compute_ctc_loss(self, x, y, reduction="mean"):
        """
        Args:
            x: log_probs, (t d)
            y: labels, (t')
        Return:
            loss
        """
        xl = torch.tensor(list(map(len, x)))
        yl = torch.tensor(list(map(len, y)))
        x = pad_sequence(x, False)  # -> (t b c)
        y = pad_sequence(y, True)  # -> (b s)
        return F.ctc_loss(x, y, xl, yl, self.blank, reduction, True)

    @staticmethod
    def mean_over_time(l):
        return torch.stack([li.mean() for li in l]).to(l[0].device)

    @property
    def nsm1_dist(self):
        device = next(self.predictor.parameters()).device
        logits = self.predictor(torch.arange(self.vocab_size).to(device))
        return Categorical(logits=logits)

    @property
    def most_probable_num_states(self):
        return (self.nsm1_dist.probs.argmax(-1) + 1).cpu()

    def compute_loss(self, x, y):
        """
        Args:
            x: videos, [(t c h w)], i.e. list of (t c h w)
            y: labels, [(t')]
        Returns:
            losses, dict of all losses, sum then and then backward
        """
        lp = self(x)
        s = [self.expand(yi) for yi in y]
        losses = {"ctc_loss": self.compute_ctc_loss(lp, s)}
        return losses

    def decode(self, prob, beam_width, prune, lm=None):
        """
        Args:
            prob: [(t d)]
            beam_width: int, number of beams
            prune: minimal probability to search
            lm: probability of the last word given the prefix
        """
        return list(self.decoder.search(prob[0], beam_width, self.blank, prune, lm))
