import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_logits(y_true, logits):
    """Slide-level metrics from a list of [n_classes] logit vectors."""
    probs = F.softmax(torch.tensor(np.array(logits)), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC": auc,
    }


def embedding_distortion(x_clean, x_adv):
    """
    Mean distortion of the targeted embeddings of one WSI.

    Returns the L2 distance, the cosine similarity and the mean absolute
    coordinate-wise difference (Feat Diff), each averaged over the targets.
    """
    diff = x_adv - x_clean
    return (
        torch.norm(diff, p=2, dim=1).mean().item(),
        F.cosine_similarity(x_clean, x_adv, dim=1).mean().item(),
        diff.abs().mean(dim=1).mean().item(),
    )
