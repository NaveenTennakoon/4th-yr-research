import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SGSResNet18(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.encoder = resnet18(True)
        self.encoder.fc = nn.Linear(512, dim)

    def forward(self, x, sgs_apply):
        """
        Args:
            x: [(t c h w)]
        Returns:
            x: [(t 512)]
        """
        xl = list(map(len, x))
        x = torch.cat(x, dim=0)
        x = sgs_apply(self.encoder, x)
        return list(x.split(xl))
