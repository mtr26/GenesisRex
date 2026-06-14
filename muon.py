"""
Muon optimizer for Genesis.

Muon (MomentUm Orthogonalized by Newton-schulz) applies Newton-Schulz
orthogonalization to the Nesterov momentum of each *matrix* parameter,
then scales the update so that the RMS step is ~1 regardless of shape.

Falls back to Adam for:
  - 1-D parameters (norms, biases, embeddings row-vectors)
  - Parameters explicitly listed in adam_params

Memory: one momentum buffer per matrix (vs Adam's two). ~33% optimizer
state savings for matrix-heavy models.

Reference: https://kellerjordan.github.io/posts/muon/
           https://arxiv.org/pdf/2502.16982  (Muon is Scalable)
"""

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz iteration (5-step quintic, ~float32 stable)
# Converges fastest with the coefficients below (Keller Jordan 2024).
# ---------------------------------------------------------------------------
def _ns5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonal polar factor of G via Newton-Schulz."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = (a * X) + ((b * A) + (c * (A @ A))) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------
class Muon(Optimizer):
    """
    Muon + AdamW in a single optimizer.

    All 2-D+ (matrix) params in `params` get Muon updates.
    All 1-D params in `params` AND all params in `adam_params` get AdamW.

    Typical split for Genesis:
        matrix_params = [p for p in model.parameters()
                         if p.ndim >= 2 and p is not model.embedding.weight]
        scalar_params  = [p for p in model.parameters()
                         if p.ndim < 2 or p is model.embedding.weight]
        optimizer = Muon(
            [{"params": matrix_params},
             {"params": scalar_params, "use_muon": False,
              "lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.1}],
            lr=learning_rate, wd=0.1,
        )

    Args:
        params:     parameter groups. Each group may override use_muon,
                    lr, wd, momentum, ns_steps, beta2, eps.
        lr:         Muon learning rate (default 0.02 — much higher than Adam).
        wd:         weight decay for both Muon and Adam groups.
        momentum:   Nesterov momentum for Muon (default 0.95).
        ns_steps:   Newton-Schulz iterations (5 is sufficient).
        adam_lr:    Adam learning rate (default 3e-4).
        adam_betas: Adam betas.
        adam_eps:   Adam epsilon.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        wd: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adam_lr: float = 3e-4,
        adam_betas: tuple = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, ns_steps=ns_steps,
            use_muon=True,
            adam_lr=adam_lr, adam_betas=adam_betas, adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", True)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2 or not use_muon:
                    self._adam_step(p, group)
                else:
                    self._muon_step(p, group)

        return loss

    def _muon_step(self, p: torch.Tensor, group: dict):
        g = p.grad
        state = self.state[p]
        if "momentum_buf" not in state:
            state["momentum_buf"] = torch.zeros_like(p)

        buf = state["momentum_buf"]
        mu = group["momentum"]
        # Nesterov momentum. Keep the stored buffer as the EMA; the look-ahead
        # gradient must not mutate it a second time.
        buf.mul_(mu).add_(g)
        g_ns = g.add(buf, alpha=mu)

        # Reshape to 2-D for NS (handles ≥3-D by merging leading dims).
        orig_shape = g_ns.shape
        G = g_ns.reshape(orig_shape[0], -1)   # [out, in*...]

        update = _ns5(G.float(), steps=group["ns_steps"]).to(dtype=p.dtype)
        update = update.reshape(orig_shape)

        # RMS scaling so step size is lr-invariant to parameter shape.
        # Kimi variant: scale by sqrt(max(1, rows/cols)).
        rows, cols = G.shape
        scale = (max(1.0, rows / cols) ** 0.5)

        # Weight decay (decoupled).
        if group["wd"] != 0.0:
            p.mul_(1.0 - group["lr"] * group["wd"])

        p.add_(update, alpha=-group["lr"] * scale)

    def _adam_step(self, p: torch.Tensor, group: dict):
        g = p.grad
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        t = state["step"]
        b1, b2 = group["adam_betas"]
        eps = group["adam_eps"]
        lr = group.get("adam_lr", group["lr"])

        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        m.mul_(b1).add_(g, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        step_size = lr * (bc2 ** 0.5) / bc1

        if group["wd"] != 0.0:
            p.mul_(1.0 - lr * group["wd"])

        p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
