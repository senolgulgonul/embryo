"""
Core components for the embryo experiments.

A parent network is observed only through input-output queries; a self-model
(surrogate ensemble) is built from those observations; a successor is then
distilled from the self-model without ever accessing the parent's parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CANDIDATES = 256
ENSEMBLE_SIZE = 5
REFIT_ITERS = 150
GROW_ITERS = 2500


# ----------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------

def truth(x):
    """Target function on [-1, 1]^2."""
    return torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])


def relative_error(model, x_test, y_true):
    """MSE normalised by the variance of the target. 1.0 = predicting the mean."""
    with torch.no_grad():
        mse = torch.mean((y_true - model(x_test)) ** 2).item()
    return mse / (torch.var(y_true).item() + 1e-8)


# ----------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------

class WideTanh(nn.Module):
    """Successor architecture. Deliberately different from the ancestor's."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class Organism(nn.Module):
    """
    A single organism: a frozen trunk, a frozen linear task head (the body,
    33 parameters), and linear self-heads reading the same trunk.

    The embedded observer has access to the body's internal representation;
    an external observer sees only behaviour.
    """

    def __init__(self, n_self_heads=ENSEMBLE_SIZE):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.task_head = nn.Linear(32, 1)
        self.self_heads = nn.ModuleList(
            [nn.Linear(32, 1) for _ in range(n_self_heads)]
        )

    def body(self, x):
        return self.task_head(self.trunk(x))

    def introspect_all(self, x):
        z = self.trunk(x)
        return torch.stack([h(z) for h in self.self_heads], dim=0)

    def freeze_body(self):
        for p in self.trunk.parameters():
            p.requires_grad = False
        for p in self.task_head.parameters():
            p.requires_grad = False


class BodyWrapper(nn.Module):
    """Exposes only the organism's behaviour, hiding its internals."""

    def __init__(self, organism):
        super().__init__()
        self.organism = organism

    def forward(self, x):
        return self.organism.body(x)


# ----------------------------------------------------------------------
# Self-model: active observation
# ----------------------------------------------------------------------

def observe(target_fn, budget, mode="covered", prior_x=None):
    """
    Build a self-model of `target_fn` from `budget` self-chosen queries.

    mode:
        "random"       take an arbitrary candidate
        "disagreement" maximise ensemble variance
        "covered"      maximise variance x distance to nearest past observation

    prior_x: inherited query history (experience inheritance). When given, the
    coverage term is computed against the lineage's accumulated observations.

    Returns (ensemble, observed_x, observed_y).
    """
    nets = [
        nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        ).to(DEVICE)
        for _ in range(ENSEMBLE_SIZE)
    ]
    opts = [optim.Adam(n.parameters(), lr=1e-2) for n in nets]

    ox = torch.zeros(0, 2, device=DEVICE)
    oy = torch.zeros(0, 1, device=DEVICE)
    hist = prior_x if prior_x is not None else torch.zeros(0, 2, device=DEVICE)

    for step in range(budget):
        cand = torch.rand(N_CANDIDATES, 2, device=DEVICE) * 2.0 - 1.0
        seen = torch.cat([hist, ox], dim=0)

        if mode == "random" or seen.shape[0] == 0:
            q = cand[0].unsqueeze(0)
        else:
            with torch.no_grad():
                preds = torch.stack([n(cand) for n in nets], dim=0)
            score = torch.var(preds, dim=0).squeeze(-1)
            if ox.shape[0] == 0:
                # ensemble untrained: disagreement is meaningless
                score = torch.ones_like(score)
            if mode == "covered":
                score = score * torch.cdist(cand, seen).min(dim=1).values
            q = cand[torch.argmax(score)].unsqueeze(0)

        with torch.no_grad():
            y = target_fn(q)

        ox = torch.cat([ox, q], dim=0)
        oy = torch.cat([oy, y], dim=0)

        for n, o in zip(nets, opts):
            for _ in range(REFIT_ITERS):
                loss = torch.mean((n(ox) - oy) ** 2)
                o.zero_grad()
                loss.backward()
                o.step()

    return nets, ox, oy


def ensemble_mean(nets, x):
    return torch.mean(torch.stack([n(x) for n in nets], dim=0), dim=0)


# ----------------------------------------------------------------------
# Birth: distil a successor
# ----------------------------------------------------------------------

def grow(nets, fx, fy, parent, iters=GROW_ITERS):
    """
    Train a successor.

    parent is not None -> inherit by distilling the parent's behaviour.
    fx is not None     -> additionally fit the successor's own observations.

    Setting parent=None gives a from-scratch learner; setting fx=None gives
    pure inheritance with no fresh information (the Born-Again control).
    """
    child = WideTanh().to(DEVICE)
    opt = optim.Adam(child.parameters(), lr=1e-3)

    for _ in range(iters):
        x = torch.rand(256, 2, device=DEVICE) * 2.0 - 1.0
        with torch.no_grad():
            target = parent(x) if parent is not None else ensemble_mean(nets, x)

        loss = torch.mean((child(x) - target) ** 2)
        if fx is not None:
            loss = loss + torch.mean((child(fx) - fy) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    for p in child.parameters():
        p.requires_grad = False

    return child
