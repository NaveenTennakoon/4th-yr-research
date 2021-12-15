import copy
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from collections import Counter

from .sgs import SGSResNet18, create_sgs_applier
from .xfmr import TransformerEncoder
from .dec import Decoder


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_num_states,
        dim=512,
        rdim=32,
        p_detach=0.75,
        rpe_k=8,
        use_sfl=True,
        ent_coef=lambda: 1.0,
        heads=4,
        semantic_layers=2,
        dropout=0.1,
        monte_carlo_samples=32,
    ):
        """
        Args:
            vocab_size: vocabulary size of the dataset.
            max_num_states: max number of state per gloss.
            dim: hidden dim for transformer encoder.
            rdim: hidden dim for the state number predictor and baseline.
            p_detach: gradient stopping proportion.
            rpe_k: the window size (one side) for relative postional encoding.
            use_sfl: whether to use stochastic fine-grained labeling.
            ent_coef: entropy loss coefficient, larger the predictor converges slower.
            heads: number of heads for transformer encoder.
            semantic_layers: number of layers for transformer encoder.
            dropout: p_dropout.
            monte_carlo_samples: number of Monte Carlo sampling for stochastic fine-grained labeling.
        """
        super().__init__()
        self.use_sfl = use_sfl
        self.monte_carlo_samples = monte_carlo_samples
        self.ent_coef = ent_coef
        self.p_detach = p_detach

        self.max_num_states = max_num_states
        self.vocab_size = vocab_size

        self.ff_visual = SGSResNet18(
            dim,
            p_detach,
        )

        self.lf_visual = SGSResNet18(
            dim,
            p_detach,
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

        if use_sfl:
            self.predictor = nn.Sequential(
                nn.Embedding(vocab_size, rdim),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, max_num_states),
            )

            self.baseline = nn.Sequential(
                nn.Embedding(vocab_size, rdim),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, 1),
            )

    def forward(self, x1, x2):
        """
        Args:
            x1: list of (t c h w)
            x2: list of (t c h w)
        Return:
            log probs [(t n)]
        """
        xl = list(map(len, x1))
        sgs_apply = create_sgs_applier(self.p_detach, xl)
        x1 = self.ff_visual(x1, sgs_apply)
        x2 = self.lf_visual(x2, sgs_apply)
        x1 = self.semantic(x1, None)
        x2 = self.semantic(x2, None)
        xout = self.semantic(x1, x2)
        x1 = torch.cat(x1)
        x2 = torch.cat(x2)
        xout = torch.cat(xout)
        x1 = self.classifier(x1)
        x2 = self.classifier(x2)
        xout = self.classifier(xout)
        x1 = x1.log_softmax(dim=-1)
        x2 = x2.log_softmax(dim=-1)
        xout = xout.log_softmax(dim=-1)
        x1 = x1.split(xl)
        x2 = x2.split(xl)
        xout = xout.split(xl)
        return xout, x1, x2

    def expand(self, y, n=None):
        """Expand to tensor"""
        return torch.tensor(self.decoder.expand(y, n)).to(y.device)

    def compute_ctc_loss(self, xout, x1, x2, y, reduction="mean"):
        """
        Args:
            x1: log_probs, (t d)
            x2: log_probs, (t d)
            y: labels, (t')
        Return:
            loss
        """
        xl = torch.tensor(list(map(len, x1)))
        yl = torch.tensor(list(map(len, y)))
        xout = pad_sequence(xout, False)  # -> (t b c)
        x1 = pad_sequence(x1, False)  # -> (t b c)
        x2 = pad_sequence(x2, False)  # -> (t b c)
        y = pad_sequence(y, True)  # -> (b s)
        loss1 = F.ctc_loss(xout, y, xl, yl, self.blank, reduction, True)
        loss2 = F.ctc_loss(x1, y, xl, yl, self.blank, reduction, True)
        loss3 = F.ctc_loss(x2, y, xl, yl, self.blank, reduction, True)
        avg_loss = (loss1 + loss2 + loss3) / 3
        return avg_loss

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

    def compute_loss(self, x1, x2, y):
        """
        Args:
            x1: videos, [(t c h w)], i.e. list of (t c h w)
            x2: videos, [(t c h w)], i.e. list of (t c h w)
            y: labels, [(t')]
        Returns:
            losses, dict of all losses, sum then and then backward
        """
        lout, lp1, lp2 = self(x1, x2)
        s = [self.expand(yi) for yi in y]
        losses = {"ctc_loss": self.compute_ctc_loss(lout, lp1, lp2, s)}
        return losses

    def decode(self, prob, beam_width, prune, lm=None, nj=8):
        """
        Args:
            prob: [(t d)]
            beam_width: int, number of beams
            prune: minimal probability to search
            lm: probability of the last word given the prefix
            nj: number of jobs
        """
        if self.use_sfl:
            self.decoder.set_num_states(
                {i: int(n) for i, n in enumerate(self.most_probable_num_states)}
            )

        with Pool(nj) as pool:
            return list(
                pool.imap(
                    partial(
                        self.decoder.search,
                        beam_width=beam_width,
                        blank=self.blank,
                        prune=prune,
                        lm=lm,
                    ),
                    tqdm.tqdm(prob, "Decoding ..."),
                )
            )
