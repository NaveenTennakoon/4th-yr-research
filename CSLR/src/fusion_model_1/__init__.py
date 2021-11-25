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

from .visual import SGSResNet18
from .sgs import create_sgs_applier
from .xfmr import TransformerEncoder
from .mc_xfmr import MC_TransformerEncoder
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
        add_ratio=1.0
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
            add_ratio: ratio of second stream considered with main stream.
        """
        super().__init__()
        self.use_sfl = use_sfl
        self.monte_carlo_samples = monte_carlo_samples
        self.ent_coef = ent_coef
        self.p_detach = p_detach

        self.max_num_states = max_num_states
        self.vocab_size = vocab_size

        self.ff_visual = SGSResNet18(dim)
        self.lf_visual = SGSResNet18(dim)

        self.f_semantic = TransformerEncoder(
            dim,
            heads,
            semantic_layers,
            dropout,
            rpe_k,
        )

        self.l_semantic = TransformerEncoder(
            dim,
            heads,
            semantic_layers,
            dropout,
            rpe_k,
        )

        self.mcf_semantic = MC_TransformerEncoder(
            dim,
            heads,
            semantic_layers,
            dropout,
            rpe_k,
        )

        self.mcl_semantic = MC_TransformerEncoder(
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
        self.classifier = nn.Linear(dim*2, self.decoder.total_states + 1, bias=False)

        self.blank = self.decoder.total_states  # last dim as blank

    def forward(self, x1, x2):
        """
        Args:
            x: list of (t c h w)
        Return:
            log probs [(t n)]
        """
        xl = list(map(len, x1))

        sgs_apply = create_sgs_applier(self.p_detach, xl)

        x1 = self.ff_visual(x1, sgs_apply)
        x1 = self.f_semantic(x1)

        x2 = self.lf_visual(x2, sgs_apply)
        x2 = self.l_semantic(x2)

        x1_out = self.mcf_semantic(x1, x2)
        x2_out = self.mcl_semantic(x2, x1)

        x1_out = torch.cat(x1_out)
        x2_out = torch.cat(x2_out)
        x = torch.cat([x1_out, x2_out], dim=1)

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

    def compute_loss(self, x1, x2, y):
        """
        Args:
            x1: ff_videos, [(t c h w)], i.e. list of (t c h w)
            x2: lf_videos, [(t c h w)], i.e. list of (t c h w)
            y: labels, [(t')]
        Returns:
            losses, dict of all losses, sum then and then backward
        """
        lpi = self(x1, x2)
        s = [self.expand(yi) for yi in y]
        losses = {"ctc_loss": self.compute_ctc_loss(lpi, s)}

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