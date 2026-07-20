# embryo

Can a neural network model another network from behaviour alone, produce a
successor from that model, and can a chain of such successors accumulate
knowledge?

This repository contains a small, fully controlled study of that question.
A parent network is observed only through input–output queries; a self-model
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
| 2 | Access to a system's internal representation confers no advantage until the observation budget reaches the point where the system becomes determined. The body has 33 parameters; between budgets 16 and 33 the embedded observer's error drops 25×, then the advantage closes again at large budgets. | see `exp2_threshold.py` |
| 3 | Ensemble disagreement alone is indistinguishable from random sampling as an acquisition rule. Disagreement × coverage is significantly better, mainly by eliminating catastrophic runs (IQR falls 3×). | 21/50, p = 0.48 (disagreement vs random); 38/50, p < 0.0001 (covered) |
| 4 | Inheriting the parent's *query history* as well as its knowledge cuts the gap between a lineage and a single undivided learner from 6.6× to 2.4×. Roughly two thirds of the cost of splitting a budget across generations is untransmitted experience, not imperfect knowledge transfer. | 20/20, p < 0.0001 |

Two preliminary findings were **retracted** when statistical power was
increased; both are documented in the paper rather than removed.

## Layout

```
embryo.py                        core: ground truth, architectures, observation, birth
experiments/
  exp1_accumulation.py           finding 1
  exp2_threshold.py              finding 2
  exp3_acquisition.py            finding 3
  exp4_experience.py             finding 4
paper/                           LaTeX source and PDF
```

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
  same budget.

## Citation

Paper: `paper/main.pdf`. arXiv link to follow.

## License

MIT
