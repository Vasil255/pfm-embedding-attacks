"""
Attacks the patch embeddings of the test WSIs and measures the slide-level degradation.

    python attack.py --pfm CONCH
    python attack.py --pfm UNI2 --aggregators ABMIL --epsilons 0.15 --ratios 0.1

For each WSI and selection ratio rho, M = max(1, floor(rho * N)) patches are drawn
at random. The subset and the attack seed depend only on the WSI and rho, so every
budget and every attack perturbs the same patches from the same random start.

Loads <output_dir>/models/<PFM>_<aggregator>/model.pth and writes one row per
(attack, eps, rho), plus the clean baseline, to <output_dir>/attacks/<PFM>_<aggregator>.csv.
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.attacks import pgd_attack, random_attack, rfgsm_attack
from src.data import CLASSES, PFMS, load_split
from src.models import AGGREGATORS, build_model
from src.utils import embedding_distortion, evaluate_logits, set_random_seeds

ROOT = os.path.dirname(os.path.abspath(__file__))

ATTACKS = ["RANDOM", "RFGSM", "PGD"]
EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def run_attacks(model, bags, labels, attacks, epsilons, ratios, steps, seed, device):
    """Returns the clean baseline and one row per (attack, eps, rho), averaged over the WSIs."""
    model.eval()

    clean_logits = []
    with torch.no_grad():
        for bag in bags:
            clean_logits.append(model(torch.as_tensor(bag, dtype=torch.float32).to(device).unsqueeze(0)).cpu().numpy()[0])
    clean = evaluate_logits(labels, clean_logits)
    clean_correct = np.argmax(clean_logits, axis=1) == labels

    configs = [(atk, eps, ratio) for atk in attacks for eps in epsilons for ratio in ratios]
    adv_logits = {c: [] for c in configs}
    distortion = {c: [] for c in configs}
    elapsed = {c: 0.0 for c in configs}

    for i, bag_np in enumerate(tqdm(bags, desc="Attacking WSIs")):
        bag = torch.tensor(bag_np, dtype=torch.float32).to(device)
        target = torch.tensor([labels[i]], dtype=torch.long, device=device)
        n_patches = bag.shape[0]

        for ratio in ratios:
            wsi_seed = seed + i * 1000 + int(ratio * 100)
            random.seed(wsi_seed)
            idx = random.sample(range(n_patches), max(1, int(n_patches * ratio)))
            x_clean = bag[idx]

            def loss_fn(x_adv):
                # Substitutes the perturbed targets into the full bag
                attacked = bag.clone()
                attacked[idx] = x_adv
                return F.cross_entropy(model(attacked.unsqueeze(0)), target)

            for eps in epsilons:
                for atk in attacks:
                    start = time.time()
                    torch.manual_seed(wsi_seed)
                    torch.cuda.manual_seed_all(wsi_seed)

                    if atk == "RANDOM":
                        x_adv = random_attack(x_clean, eps)
                    elif atk == "RFGSM":
                        x_adv = rfgsm_attack(loss_fn, x_clean, eps)
                    else:
                        x_adv = pgd_attack(loss_fn, x_clean, eps, steps=steps)

                    attacked = bag.clone()
                    attacked[idx] = x_adv
                    with torch.no_grad():
                        adv_logits[(atk, eps, ratio)].append(model(attacked.unsqueeze(0)).cpu().numpy()[0])
                    distortion[(atk, eps, ratio)].append(embedding_distortion(x_clean, x_adv))
                    elapsed[(atk, eps, ratio)] += time.time() - start

    rows = [{"Attack": "None", "Epsilon": 0.0, "Ratio": 0.0, "Steps": 0, **clean,
             "BACC_Drop": 0.0, "ASR": 0.0, "L2_Dist": 0.0, "Cos_Sim": 1.0, "Feat_Diff": 0.0, "Time_s": 0.0}]
    for c in configs:
        atk, eps, ratio = c
        res = evaluate_logits(labels, adv_logits[c])
        flipped = clean_correct & (np.argmax(adv_logits[c], axis=1) != labels)
        l2, cos, feat = np.mean(distortion[c], axis=0)
        rows.append({
            "Attack": atk, "Epsilon": eps, "Ratio": ratio, "Steps": steps if atk == "PGD" else 1 if atk == "RFGSM" else 0,
            **res,
            "BACC_Drop": clean["BACC"] - res["BACC"],
            # Fraction of the correctly classified WSIs whose prediction the attack changes
            "ASR": flipped.sum() / max(clean_correct.sum(), 1),
            "L2_Dist": l2, "Cos_Sim": cos, "Feat_Diff": feat,
            "Time_s": round(elapsed[c], 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pfm", required=True, choices=PFMS, help="Pathology foundation model of the embeddings.")
    parser.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS), choices=list(AGGREGATORS))
    parser.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS)
    parser.add_argument("--epsilons", type=float, nargs="+", default=EPSILONS, help="Per-patch L2 budgets.")
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS, help="Fractions of patches to perturb.")
    parser.add_argument("--steps", type=int, default=10, help="PGD iterations.")
    parser.add_argument("--embeddings_root", default=os.path.join(ROOT, "data", "AI4SKIN", "embeddings"),
                        help="Folder with one subfolder of <WSI>.npy files per PFM.")
    parser.add_argument("--splits_dir", default=os.path.join(ROOT, "data", "AI4SKIN", "splits"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--seed", type=int, default=42, help="Base seed of the patch selection and the random starts.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(args.seed)

    test_data, test_labels = load_split("test", os.path.join(args.embeddings_root, args.pfm), args.splits_dir)
    L = test_data[0].shape[-1]

    os.makedirs(os.path.join(args.output_dir, "attacks"), exist_ok=True)
    for agg in args.aggregators:
        print(f"\n=== {args.pfm} + {agg} ===")
        model = build_model(agg, n_classes=len(CLASSES), L=L).to(device)
        ckpt = os.path.join(args.output_dir, "models", f"{args.pfm}_{agg}", "model.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        for p in model.parameters():
            p.requires_grad_(False)

        df = run_attacks(model, test_data, test_labels, args.attacks, args.epsilons, args.ratios,
                         args.steps, args.seed, device)
        df.insert(0, "Aggregator", agg)
        df.insert(0, "Dim", L)
        df.insert(0, "PFM", args.pfm)

        out = os.path.join(args.output_dir, "attacks", f"{args.pfm}_{agg}.csv")
        df.to_csv(out, index=False)
        print(f"Clean BACC {df.BACC[0]:.4f} -> {out}")


if __name__ == "__main__":
    main()
