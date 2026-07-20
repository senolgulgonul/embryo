"""
Experiment 1: transfer is near-lossless, but copying alone creates nothing.

Arm A: each generation inherits the parent and adds fresh observations.
Arm B: each generation inherits only (Born-Again control, no new information).

Reported in Section 4.1 / Table 1 of the paper.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import wilcoxon

from embryo import DEVICE, truth, observe, grow, relative_error

N_LINEAGES = 20
N_GENERATIONS = 4
ANCESTOR_BUDGET = 25
CHILD_BUDGET = 10
N_TEST = 4000


def main():
    anc, A, B = [], [[] for _ in range(N_GENERATIONS)], [[] for _ in range(N_GENERATIONS)]

    for lineage in range(N_LINEAGES):
        torch.manual_seed(102000 + lineage)

        x_test = torch.rand(N_TEST, 2, device=DEVICE) * 2.0 - 1.0
        y_test = truth(x_test)

        nets, ox, oy = observe(truth, ANCESTOR_BUDGET)
        ancestor = grow(nets, ox, oy, None)
        anc.append(relative_error(ancestor, x_test, y_test))

        cur_a = cur_b = ancestor
        for g in range(N_GENERATIONS):
            nets, fx, fy = observe(truth, CHILD_BUDGET)
            cur_a = grow(nets, fx, fy, cur_a)
            cur_b = grow(None, None, None, cur_b)
            A[g].append(relative_error(cur_a, x_test, y_test))
            B[g].append(relative_error(cur_b, x_test, y_test))

        print(f"lineage {lineage} done", flush=True)

    print(f"\nancestor median: {np.median(anc):.4f}")
    print("\ngen | A: inherit+fresh | B: inherit only | A<B")
    for g in range(N_GENERATIONS):
        wins = int(np.sum(np.array(A[g]) < np.array(B[g])))
        print(f"  {g+1} | {np.median(A[g]):.4f} | {np.median(B[g]):.4f} | {wins}/{N_LINEAGES}")

    for name, x, y in [
        ("A(final) vs B(final)", A[-1], B[-1]),
        ("A(final) vs ancestor", A[-1], anc),
        ("B(final) vs ancestor", B[-1], anc),
    ]:
        x, y = np.array(x), np.array(y)
        _, p = wilcoxon(x, y)
        print(f"{name}: wins {int((y > x).sum())}/{len(x)} | "
              f"median diff {np.median(y - x):+.4f} | p = {p:.4f}")


if __name__ == "__main__":
    main()
