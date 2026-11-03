"""Analytic (Wronski-style) robustness mask R0 — the network's starting point.

R0 = clamp(s * exp(-d^2 / sigma^2) - t, 0, 1),   d = |mu_ref - mu_cmp| (3x3 means)
sigma^2 = c_ms * var_ref(3x3) + c_n * noise_var(mu_ref)
optional 5x5 min, optional sqrt (VST) guide domain.

The app feeds its own R0 at inference. Its exact formula is unknown to this
module (written without reading the app), so training draws the parameters at
random per sample: the net must learn to refine *an* analytic mask from the
evidence, not memorise one formula's quirks. NOMINAL is the eval setting.
"""
import torch
import torch.nn.functional as F

from .common import win_mean

NOMINAL = dict(s=2.0, t=0.12, c_ms=1.0, c_n=1.0, min5=True, sqrt_guide=False)


def random_params(rng):
    return dict(s=float(rng.uniform(1.2, 3.0)), t=float(rng.uniform(0.0, 0.3)),
                c_ms=float(rng.choice([0.0, 0.5, 1.0])), c_n=float(rng.uniform(0.5, 4.0)),
                min5=bool(rng.random() < 0.5), sqrt_guide=bool(rng.random() < 0.3))


def analytic_r0(Ir, J, noise_a, noise_b, p):
    """Ir, J (= cmp warped by the estimated flow): [B,H,W]; noise var(v) = a v + b (grey domain)."""
    if p["sqrt_guide"]:
        mu = win_mean(Ir, 1).clamp_min(1e-6)
        nv = (noise_a * mu + noise_b) / (4.0 * mu)            # delta-method variance after sqrt
        Ir, J = Ir.clamp_min(0).sqrt(), J.clamp_min(0).sqrt()
    else:
        nv = None
    mr = win_mean(Ir, 1)
    mj = win_mean(J, 1)
    vr = (win_mean(Ir * Ir, 1) - mr * mr).clamp_min(0)
    if nv is None:
        nv = noise_a * mr.clamp_min(0) + noise_b
    sig2 = (p["c_ms"] * vr + p["c_n"] * nv).clamp_min(1e-12)
    R = (p["s"] * torch.exp(-(mr - mj) ** 2 / sig2) - p["t"]).clamp(0, 1)
    if p["min5"]:
        R = -F.max_pool2d(F.pad(-R[:, None], (2, 2, 2, 2), mode="replicate"), 5, 1)[:, 0]
    return R
