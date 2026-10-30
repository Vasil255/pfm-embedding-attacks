"""
Trains the MIL aggregators on the unit-normalized patch embeddings of one PFM.

    python train.py --pfm CONCH
    python train.py --pfm UNI2 --aggregators TransMIL --embeddings_root /path/to/embeddings

For each aggregator it writes to <output_dir>/models/<PFM>_<aggregator>/:
    model.pth     weights of the epoch with the lowest validation loss
    history.csv   training and validation loss/accuracy per epoch
    metrics.csv   validation and test metrics of the selected checkpoint
"""

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.data import CLASSES, PFMS, load_split
from src.models import AGGREGATORS, build_model
from src.utils import evaluate_logits, set_random_seeds

ROOT = os.path.dirname(os.path.abspath(__file__))


def predict(model, bags, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for bag in bags:
            logits.append(model(torch.as_tensor(bag, dtype=torch.float32).to(device)).cpu().numpy()[0])
    return logits


def train(model, optimizer, scheduler, criterion, train_data, train_labels, val_data, val_labels, epochs, device):
    """
    Trains with a batch of one WSI and keeps the weights of the epoch with the
    lowest validation loss. A new minimum must improve on the previous one by
    at least 1e-3.
    """
    history = []
    best_loss, best_weights = None, None

    for epoch in range(epochs):
        combined = list(zip(train_data, train_labels))
        random.shuffle(combined)
        train_data, train_labels = zip(*combined)

        model.train()
        t_loss, t_acc = 0.0, 0
        for bag, label in zip(train_data, train_labels):
            optimizer.zero_grad()
            x = torch.as_tensor(bag, dtype=torch.float32).to(device)
            y = torch.tensor([label], dtype=torch.long).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += (logits.argmax(dim=-1) == y).sum().item()

        model.eval()
        v_loss, v_acc = 0.0, 0
        with torch.no_grad():
            for bag, label in zip(val_data, val_labels):
                x = torch.as_tensor(bag, dtype=torch.float32).to(device)
                y = torch.tensor([label], dtype=torch.long).to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item()
                v_acc += (logits.argmax(dim=-1) == y).sum().item()

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": t_loss / len(train_data), "train_acc": t_acc / len(train_data),
            "val_loss": v_loss / len(val_data), "val_acc": v_acc / len(val_data),
        })
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"train loss {history[-1]['train_loss']:.4f} acc {history[-1]['train_acc']:.4f} | "
              f"val loss {history[-1]['val_loss']:.4f} acc {history[-1]['val_acc']:.4f}")

        val_loss = history[-1]["val_loss"]
        if best_loss is None or val_loss <= best_loss - 1e-3:
            best_loss, best_weights = val_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pfm", required=True, choices=PFMS, help="Pathology foundation model of the embeddings.")
    parser.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS), choices=list(AGGREGATORS))
    parser.add_argument("--embeddings_root", default=os.path.join(ROOT, "data", "AI4SKIN", "embeddings"),
                        help="Folder with one subfolder of <WSI>.npy files per PFM.")
    parser.add_argument("--splits_dir", default=os.path.join(ROOT, "data", "AI4SKIN", "splits"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(args.seed)

    embeddings_dir = os.path.join(args.embeddings_root, args.pfm)
    train_data, train_labels = load_split("train", embeddings_dir, args.splits_dir)
    val_data, val_labels = load_split("val", embeddings_dir, args.splits_dir)
    test_data, test_labels = load_split("test", embeddings_dir, args.splits_dir)
    L = train_data[0].shape[-1]
    print(f"{args.pfm}: d = {L} | train {len(train_data)} | val {len(val_data)} | test {len(test_data)} WSIs")

    # Cross-entropy weighted by inverse training-class frequencies
    class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for agg in args.aggregators:
        print(f"\n=== {args.pfm} + {agg} ===")
        # Same RNG state for every aggregator, whatever the order they run in
        set_random_seeds(args.seed)

        model = build_model(agg, n_classes=len(CLASSES), L=L).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

        history = train(model, optimizer, scheduler, criterion, train_data, train_labels,
                        val_data, val_labels, args.epochs, device)

        out_dir = os.path.join(args.output_dir, "models", f"{args.pfm}_{agg}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
        history.to_csv(os.path.join(out_dir, "history.csv"), index=False)

        metrics = pd.DataFrame([
            {"split": "val", **evaluate_logits(val_labels, predict(model, val_data, device))},
            {"split": "test", **evaluate_logits(test_labels, predict(model, test_data, device))},
        ])
        metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
