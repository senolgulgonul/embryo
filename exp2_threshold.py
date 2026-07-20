"""
Experiment 2: introspective access is a budget-dependent threshold phenomenon.

An embedded observer (linear heads reading the body's own trunk) is compared
against an external observer (behaviour only) at equal query budgets. The body
is a 32-input linear head, i.e. 33 unknowns.

Reported in Section 4.2 / Table 2 of the paper.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import wilcoxon

from embryo import (DEVICE, N_CANDIDATES, Organism, BodyWrapper,
                    observe, ensemble_mean)

BUDGETS = [8, 16, 33, 64]
N_ORGANISMS = 30
N_TEST = 4000
FIT_ITERS = 200


def train_self_heads(org, budget):
    opt = optim.Adam(org.self_heads.parameters(), lr=1e-2)
    ox = torch.zeros(0, 2, device=DEVICE)
    oy = torch.zeros(0, 1, device=DEVICE)

    for step in range(budget):
        cand = torch.rand(N_CANDIDATES, 2, device=DEVICE) * 2.0 - 1.0
        if step > 0:
            with torch.no_grad():
                preds = org.introspect_all(cand)
            score = torch.var(preds, dim=0).squeeze(-1)
            score = score * torch.cdist(cand, ox).min(dim=1).values
            q = cand[torch.argmax(score)].unsqueeze(0)
        else:
            q = cand[0].unsqueeze(0)

        with torch.no_grad():
            y = org.body(q)
        ox = torch.cat([ox, q], dim=0)
        oy = torch.cat([oy, y], dim=0)

        for _ in range(FIT_ITERS):
            loss = torch.mean((org.introspect_all(ox) - oy.unsqueeze(0)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    emb = {b: [] for b in BUDGETS}
    ext = {b: [] for b in BUDGETS}

    for budget in BUDGETS:
        for i in range(N_ORGANISMS):
            torch.manual_seed(105000 + i)      # paired across budgets

            org = Organism().to(DEVICE)
            org.freeze_body()

            x_test = torch.rand(N_TEST, 2, device=DEVICE) * 2.0 - 1.0
            with torch.no_grad():
                y_body = org.body(x_test)
            var = torch.var(y_body).item() + 1e-8

            train_self_heads(org, budget)
            with torch.no_grad():
                pred = torch.mean(org.introspect_all(x_test), dim=0)
            emb[budget].append(
                torch.mean((y_body - pred) ** 2).item() / var
            )

            nets, _, _ = observe(BodyWrapper(org), budget, mode="covered")
            with torch.no_grad():
                pred = ensemble_mean(nets, x_test)
            ext[budget].append(
                torch.mean((y_body - pred) ** 2).item() / var
            )

        print(f"budget {budget} done", flush=True)

    print("\nbudget | embedded | external | ratio | wins | p")
    for b in BUDGETS:
        e, x = np.array(emb[b]), np.array(ext[b])
        _, p = wilcoxon(e, x)
        print(f"{b:6d} | {np.median(e):.5f} | {np.median(x):.5f} | "
              f"{np.median(x)/np.median(e):6.2f} | "
              f"{int((e < x).sum())}/{N_ORGANISMS} | {p:.4f}")


if __name__ == "__main__":
    main()
