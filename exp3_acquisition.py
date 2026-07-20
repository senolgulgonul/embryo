"""
Experiment 3: disagreement alone is not a useful acquisition criterion.

Three rules compared at ancestor level on paired seeds: random,
disagreement, and disagreement x coverage.

Reported in Section 4.3 / Table 3 of the paper.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import wilcoxon

from embryo import DEVICE, truth, observe, grow, relative_error

MODES = ["disagreement", "random", "covered"]
N_RUNS = 50
BUDGET = 25
N_TEST = 4000


def main():
    err = {m: [] for m in MODES}
    nnd = {m: [] for m in MODES}

    for run in range(N_RUNS):
        x_test = torch.rand(N_TEST, 2, device=DEVICE) * 2.0 - 1.0
        y_test = truth(x_test)

        for mode in MODES:
            torch.manual_seed(101000 + run)   # paired: same seed per mode
            nets, ox, oy = observe(truth, BUDGET, mode=mode)
            model = grow(nets, ox, oy, None)
            err[mode].append(relative_error(model, x_test, y_test))

            d = torch.cdist(ox, ox)
            d.fill_diagonal_(9.0)
            nnd[mode].append(d.min(dim=1).values.mean().item())

        if run % 10 == 0:
            print(f"run {run} done", flush=True)

    print("\nrule          | median | IQR    | mean   | nn dist")
    for mode in MODES:
        e = np.array(err[mode])
        iqr = np.percentile(e, 75) - np.percentile(e, 25)
        print(f"{mode:13s} | {np.median(e):.4f} | {iqr:.4f} | "
              f"{e.mean():.4f} | {np.median(nnd[mode]):.3f}")

    print()
    for a, b in [("disagreement", "random"), ("covered", "random"),
                 ("covered", "disagreement")]:
        x, y = np.array(err[a]), np.array(err[b])
        _, p = wilcoxon(x, y)
        print(f"{a} vs {b}: wins {int((y > x).sum())}/{N_RUNS} | "
              f"median diff {np.median(y - x):+.4f} | p = {p:.4f}")


if __name__ == "__main__":
    main()
