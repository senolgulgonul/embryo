# embryo

Can a neural network model another network from behaviour alone, produce a
successor from that model, and can a chain of such successors accumulate
knowledge?

This repository contains a small, fully controlled study of that question.
A parent network is observed only through input-output queries; a self-model
(a surrogate ensemble) is built from those observations; a successor of a
*different* architecture is then distilled from the self-model without ever
accessing the parent's parameters.

Everything runs in a couple of hours on a single GPU.

## Findings

All results use a paired-seed design and Wilcoxon signed-rank tests.
Metric: relative error = MSE / Var(target), so 1.0 = predicting the mean.

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Behavioural transfer is near-lossless, but copying alone creates nothing. A lineage improves only when each successor adds fresh observations; pure inheritance is statistically indistinguishable from the ancestor. | 20/20, p < 0.0001 (fresh vs inherit-only); p = 0.26 (inherit-only vs ancestor) |
| 2 | Access to a system's internal representation confers no advantage until the observation budget reaches the parameter count of the mechanism (33 here). At that point the embedded observer's error drops 25x, and the advantage keeps widening: at budget 128 the ratio is 206x. The threshold tracks the parameter count across trunk widths and input dimensions. | n = 30; 30/30 wins from budget 33 on, p < 0.0001 |
| 3 | Ensemble disagreement alone is indistinguishable from random sampling as an acquisition rule. Disagreement x coverage is significantly better, mainly by eliminating catastrophic runs (IQR falls 3x). | 21/50, p = 0.48 (disagreement vs random); 38/50, p < 0.0001 (covered) |
| 4 | Inheriting the parent's *query history* as well as its knowledge cuts the gap between a lineage and a single undivided learner from 6.6x to 2.4x. Roughly two thirds of the cost of splitting a budget across generations is untransmitted experience, not imperfect knowledge transfer. | 20/20, p < 0.0001 |

A five-target replication bounds finding 1: accumulation is strong on
mid-difficulty targets (product, saddle) and vanishes when the ancestor is
already at the attainable ceiling, whether that ceiling is near zero (ridge)
or far from it (radial, high-frequency). Inheritance-only lineages never beat
the ancestor on any target, and visibly decay on the two hardest ones.

Three preliminary findings were **retracted** when statistical power was
increased; all three are documented in the paper rather than removed, and the
runs behind them are preserved in the development notebook below.

## Layout

```
embryo.py                 core: ground truth, architectures, observation, birth
embryo_notebook.ipynb     full development record (28 cells, see mapping below)
experiments/
  exp1_accumulation.py    finding 1
  exp2_threshold.py       finding 2 (main sweep, budgets 8-64)
  exp3_acquisition.py     finding 3
  exp4_experience.py      finding 4
```

The scripts are cleaned, self-contained versions for quick replication. The
notebook is the primary record: development happened there, cell by cell, and
it contains every experiment in the paper, including the ones that were later
retracted.

## Notebook to paper mapping

| Paper | Notebook cell |
|-------|---------------|
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

## Running

```bash
pip install -r requirements.txt
python experiments/exp1_accumulation.py
```

In Colab:

```python
!git clone https://github.com/senolgulgonul/embryo.git
%cd embryo
!python experiments/exp3_acquisition.py
```

Each script prints its own table and significance tests. Seeds are fixed, so
runs are reproducible up to GPU non-determinism.

## Design notes

- The successor never sees the parent's weights. Inheritance is behavioural,
not genetic; a WideTanh successor is distilled from a ReLU-based self-model
precisely so that transfer cannot be a disguised weight copy.
- `grow(parent=None)` gives a from-scratch learner and `grow(fx=None)` gives
pure inheritance with no new information. These two controls are what
separate genuine accumulation from the Born-Again distillation effect.
- The organism in `exp2_threshold.py` has a frozen trunk and a frozen linear
task head (33 parameters). The embedded observer reads the same trunk; the
external one sees only behaviour. Both use the same acquisition rule and the
same budget. In the threshold experiments the object being modelled is the
organism's own randomly initialised body, so the ground truth function plays
no role there.

## Citation

Paper preprint will be linked here; arXiv submission pending.

## License

MIT
