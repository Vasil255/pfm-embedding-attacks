import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Order of the GT indices in the split CSVs
CLASSES = ["lm", "lms", "df", "dfs", "scm", "afx"]
CLASS_NAMES = {
    "lm": "leiomyoma",
    "lms": "leiomyosarcoma",
    "df": "dermatofibroma",
    "dfs": "dermatofibrosarcoma",
    "scm": "spindle cell melanoma",
    "afx": "atypical fibroxanthoma",
}

PFMS = ["CONCH", "KEEP", "GPFM", "KAIKO", "VIRCHOW2", "UNI2"]


def load_split(split, embeddings_dir, splits_dir):
    """
    Loads the bags of one split as unit-normalized patch embeddings.

    Args:
        split (str): 'train', 'val' or 'test'.
        embeddings_dir (str): Folder with one <WSI>.npy array of shape [N, d] per slide.
        splits_dir (str): Folder with the split CSVs (columns 'WSI' and 'GT').

    Returns:
        tuple: List of [N, d] float32 arrays, and their labels as an int64 array.
    """
    df = pd.read_csv(os.path.join(splits_dir, f"{split}.csv"))

    bags = []
    for wsi in tqdm(df["WSI"], desc=f"Loading {split}", leave=False):
        path = os.path.join(embeddings_dir, f"{wsi}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding for {wsi}: {path}")
        h = torch.tensor(np.load(path), dtype=torch.float32)
        bags.append(F.normalize(h, p=2, dim=-1).numpy())

    return bags, df["GT"].to_numpy(dtype=np.int64)
