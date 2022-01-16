import torch
import torch.nn as nn
from torchvision.models import resnet18

class SGSResNet18(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.encoder = resnet18(True)
        self.encoder.fc = nn.Linear(512, dim)

    def forward(self, x):
        """
        Args:
            x: [(t c h w)]
        Returns:
            x: [(t 512)]
        """
        xl = list(map(len, x))
        x = torch.cat(x, dim=0)
        x = self.encoder(x)
        return list(x.split(xl))
