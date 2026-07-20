"""
Experiment 4: the cost of untransmitted experience.

Successors inherit what the parent knows (distillation) but, by default, not
where the parent looked. Here the coverage term is computed against the
lineage's accumulated query history.

Reported in Section 4.4 / Table 4 of the paper.
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
    res = {False: [[] for _ in range(N_GENERATIONS + 1)],
           True: [[] for _ in range(N_GENERATIONS + 1)]}

    for lineage in range(N_LINEAGES):
        x_test = torch.rand(N_TEST, 2, device=DEVICE) * 2.0 - 1.0
        y_test = truth(x_test)

        for inherit_experience in [False, True]:
            torch.manual_seed(103000 + lineage)   # paired

            nets, ox, oy = observe(truth, ANCESTOR_BUDGET)
            cur = grow(nets, ox, oy, None)
            history = ox.clone()
            res[inherit_experience][0].append(relative_error(cur, x_test, y_test))

            for g in range(N_GENERATIONS):
                nets, fx, fy = observe(
                    truth, CHILD_BUDGET,
                    prior_x=history if inherit_experience else None,
                )
                cur = grow(nets, fx, fy, cur)
                if inherit_experience:
                    history = torch.cat([history, fx], dim=0)
                res[inherit_experience][g + 1].append(
                    relative_error(cur, x_test, y_test)
                )

        print(f"lineage {lineage} done", flush=True)

    print("\ngen | no experience | with experience | with<without")
    for g in range(N_GENERATIONS + 1):
        a, b = np.array(res[False][g]), np.array(res[True][g])
        label = "anc" if g == 0 else f"  {g}"
        print(f"{label} | {np.median(a):.4f} | {np.median(b):.4f} | "
              f"{int((b < a).sum())}/{N_LINEAGES}")

    a, b = np.array(res[False][-1]), np.array(res[True][-1])
    _, p = wilcoxon(a, b)
    print(f"\nfinal generation: median diff {np.median(a - b):+.4f} | p = {p:.4f}")


if __name__ == "__main__":
    main()
