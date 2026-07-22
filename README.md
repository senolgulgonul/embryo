# embryo

Can a neural network model another network from behaviour alone, produce a
successor from that model, and can a chain of such successors accumulate
knowledge? And once a lineage exists, when does generational turnover actually
pay, and what property of a learner pays for it?

This repository contains two small, fully controlled studies of those
questions. A parent network is observed only through input-output queries; a
self-model (a surrogate ensemble) is built from those observations; a successor
of a different architecture is then distilled from the self-model without ever
accessing the parent's parameters. The first paper establishes the lineage and
its transfer properties; the second asks what generational turnover is worth
once the world can change.

Everything runs in a couple of hours on a single GPU. All results use a
paired-seed design and Wilcoxon signed-rank tests. Metric: relative error =
MSE / Var(target), so 1.0 = predicting the mean.

---

## Paper 1: Embryo — Self-Directed Behavioural Modeling and Generational Transfer

Manuscript under review, Artificial Life.

### Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | Behavioural transfer is near-lossless, but copying alone creates nothing. A lineage improves only when each successor adds fresh observations; pure inheritance is statistically indistinguishable from the ancestor. | 20/20, p < 0.0001 (fresh vs inherit-only); p = 0.26 (inherit-only vs ancestor) |
| 2 | Access to a system's internal representation confers no advantage until the observation budget reaches the parameter count of the mechanism (33 here). At that point the embedded observer's error drops 25x, and the advantage keeps widening: at budget 128 the ratio is 206x. The threshold tracks the parameter count across trunk widths and input dimensions. | n = 30; 30/30 wins from budget 33 on, p < 0.0001 |
| 3 | Ensemble disagreement alone is indistinguishable from random sampling as an acquisition rule. Disagreement x coverage is significantly better, mainly by eliminating catastrophic runs (IQR falls 3x). | 21/50, p = 0.48 (disagreement vs random); 38/50, p < 0.0001 (covered) |
| 4 | Inheriting the parent's query history as well as its knowledge cuts the gap between a lineage and a single undivided learner from 6.6x to 2.4x. Roughly two thirds of the cost of splitting a budget across generations is untransmitted experience, not imperfect knowledge transfer. | 20/20, p < 0.0001 |

A five-target replication bounds finding 1: accumulation is strong on
mid-difficulty targets (product, saddle) and vanishes when the ancestor is
already at the attainable ceiling, whether that ceiling is near zero (ridge) or
far from it (radial, high-frequency). Inheritance-only lineages never beat the
ancestor on any target, and visibly decay on the two hardest ones.

Three preliminary findings were retracted when statistical power was increased;
all three are documented in the paper rather than removed, and the runs behind
them are preserved in the development notebook.

### Layout

```
embriyo1_20260719_1012.ipynb   canonical record: every experiment, cell by cell
embryo.py                      core: ground truth, architectures, observation, birth
experiments/
  exp1_accumulation.py    finding 1
  exp2_threshold.py       finding 2 (main sweep, budgets 8-64)
  exp3_acquisition.py     finding 3
  exp4_experience.py      finding 4
requirements.txt
```

The notebook is the canonical source. Development happened there, cell by cell,
and it contains every experiment in both papers, including the runs behind the
retracted findings and all reported numbers. The scripts under `experiments/`
are cleaned, self-contained extractions of the four main experiments, provided
as an optional way to replicate a single finding quickly without opening the
full notebook; where a script and the notebook disagree, the notebook is
authoritative.

### Notebook to paper mapping

| Paper | Notebook cell |
|---|---|
| Table 1 (accumulation, arms A/B/C) | Cell 21 |
| Table 2 (embedded vs external, budgets 8-64 and 128) | Cells 23 and 24 |
| Table 3 (threshold scaling with parameter count) | Cell 25 |
| Table 4 (linearity decomposition) | Cell 26 |
| Table 5 (acquisition rules, n = 50) | Cell 20 (code preserved; output not stored, rerun takes under an hour on a T4) |
| Table 6 (experience inheritance) | Cell 22 |
| Table 7 (five-target replication) | Cell 27 |
| Retracted: acquisition headlines | Cells 18 and 19, resolved by Cell 20 |
| Retracted: validation leak | Cells 14 and 15, corrected in Cell 16 |
| Retracted: advantage closing at budget 128 | Cell 12 (n = 20 preliminary), reversed by Cells 23 and 24 |

---

## Paper 2: Resynthesis — Generational Turnover and the Value of Forgetting

Adaptive Behavior, Manuscript ID AB-26-0306, submitted 2026-07-22.

Given the lineage of Paper 1, this study asks when generational turnover earns
its cost. In a fixed world a single undivided learner beats a lineage on the
same budget, and no archive architecture recovers the gap. When the world
changes the ordering reverses, and a single long-lived learner that merely
forgets its own stale observations, without ever being replaced, captures the
advantage otherwise attributed to turnover. The epistemic value of turnover
reduces to the forgetting that replacement enforces.

### Files

- `embriyo1_20260719_1012.ipynb` — full notebook with embedded outputs;
  canonical source for the fixed-world and phase-drift results.
- `results/embryo3_churn_results_20260721_1954.json` — per-lineage churn and
  composition results (20 lineages each for I_archive, I_window, I_oracle,
  P_lineage, plus the ancestor). Source for Table 4, Table 5 and Figure 3.
- `results/churn_results_20260721_1954.txt` — human-readable console dump of the
  same churn medians and paired win counts (EXP1 and EXP2).

### Which table/figure comes from where

| Manuscript item | Source |
|---|---|
| Table 1, Table 2, Table 3 | notebook `embriyo1_20260719_1012.ipynb` |
| Figure 1, Figure 2 | notebook `embriyo1_20260719_1012.ipynb` |
| Table 4, Figure 3 | `results/embryo3_churn_results_20260721_1954.json` (EXP1, c = 0.0 to 0.4) |
| Table 5 | `results/embryo3_churn_results_20260721_1954.json` (EXP2, all-flip vs all-replace) |

The medians and paired win counts reported in the manuscript were recomputed
from the JSON and match exactly.

---

## Running

```
pip install -r requirements.txt
python experiments/exp1_accumulation.py
```

In Colab:

```
!git clone https://github.com/senolgulgonul/embryo.git
%cd embryo
!python experiments/exp3_acquisition.py
```

Each script prints its own table and significance tests. Seeds are fixed, so
runs are reproducible up to GPU non-determinism. Running the scripts is
optional; the notebook reproduces everything on its own.

## Design notes

- The successor never sees the parent's weights. Inheritance is behavioural,
  not genetic; a WideTanh successor is distilled from a ReLU-based self-model
  precisely so that transfer cannot be a disguised weight copy.
- `grow(parent=None)` gives a from-scratch learner and `grow(fx=None)` gives
  pure inheritance with no new information. These two controls are what separate
  genuine accumulation from the Born-Again distillation effect.
- The organism in `exp2_threshold.py` has a frozen trunk and a frozen linear
  task head (33 parameters). The embedded observer reads the same trunk; the
  external one sees only behaviour. Both use the same acquisition rule and the
  same budget.

## Citation

Paper 1 preprint will be linked here; arXiv submission pending.
Paper 2: Adaptive Behavior, Manuscript ID AB-26-0306.

## License

MIT
