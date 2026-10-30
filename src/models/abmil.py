import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """
    Gated attention-based MIL (Ilse et al., ICML 2018).

    Maps a bag of patch embeddings [N, L] to slide logits [1, n_classes]. The
    slide representation z is L2-normalized before the linear classifier.
    """

    def __init__(self, n_classes, L):
        super().__init__()
        self.L = L
        self.D = int(self.L / 4)
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Linear(self.L, n_classes)

    def forward(self, h):
        if h.dim() == 3:
            h = h.squeeze(0)

        # Gated attention over the N patches
        a = torch.softmax(self.attention_weights(self.attention_V(h) * self.attention_U(h)), dim=0)  # [N, 1]

        # Attention-weighted sum of the patch embeddings
        z = torch.mm(h.transpose(1, 0), a).transpose(1, 0)  # [1, L]
        z = F.normalize(z, p=2, dim=1)

        return self.classifier(z)
