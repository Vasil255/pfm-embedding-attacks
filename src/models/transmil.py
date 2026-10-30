import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator: depth-wise convolutions at three scales."""

    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


class TransMIL(nn.Module):
    """
    Transformer-based MIL (Shao et al., NeurIPS 2021).

    Maps a bag of patch embeddings [N, L] to slide logits [1, n_classes]. The
    slide representation z (class token) is L2-normalized before the linear
    classifier.
    """

    def __init__(self, n_classes, L):
        super().__init__()
        self.pos_layer = PPEG(dim=L)
        # Unused, kept for reproducible initialization and checkpoint compatibility
        self._fc1 = nn.Sequential(nn.Linear(int(L * 2), L), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, L))
        self.layer1 = TransLayer(dim=L)
        self.layer2 = TransLayer(dim=L)
        self.norm = nn.LayerNorm(L)
        self.classifier = nn.Sequential(nn.Linear(L, n_classes))

    def forward(self, h):
        h = h.float()
        if h.dim() == 2:
            h = h.unsqueeze(0)

        # Pad the sequence to a square number of tokens for the PPEG
        N = h.shape[1]
        _H = _W = int(np.ceil(np.sqrt(N)))
        h = torch.cat([h, h[:, :_H * _W - N, :]], dim=1)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        z = self.norm(h)[:, 0]
        z = F.normalize(z, p=2, dim=1)

        return self.classifier(z)
