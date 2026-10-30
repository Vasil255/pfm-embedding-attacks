"""
L2-bounded adversarial attacks on unit-normalized patch embeddings.

Every attack receives the M target embeddings `x_clean` [M, d] of one WSI and a
closure `loss_fn(x_adv) -> scalar` that substitutes them into the bag, runs the
frozen MIL aggregator and classifier, and returns the slide-level cross-entropy.
Each row of the returned tensor satisfies ||delta_i||_2 <= eps before being
renormalized onto the unit hypersphere.

The implementations derive from TorchAttacks (Kim, 2020), moved from the pixel
space (L_inf hypercube) to the embedding space (L2 ball on the hypersphere).
"""

import torch


def project_l2(delta, eps):
    """Pi_{L2,eps}: rescales each row of `delta` to norm eps if it exceeds it."""
    norms = torch.norm(delta, p=2, dim=-1, keepdim=True)
    factor = torch.clamp(eps / (norms + 1e-12), max=1.0)
    return delta * factor


def project_to_sphere(v, x_clean):
    """
    Rescales each row of `v` to the norm of the matching row of `x_clean`.

    The clean embeddings are unit-normalized, so this is the renormalization
    h* = (h + delta) / ||h + delta||_2 of.
    """
    radius = torch.norm(x_clean, p=2, dim=-1, keepdim=True)
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return v * (radius / v_norm)


def l2_ball_noise(x, eps):
    """Samples one perturbation per row, uniformly within the L2 ball of radius eps."""
    noise = torch.randn_like(x)
    direction = noise / torch.norm(noise, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    # The d-th root of a uniform radius spreads the samples over the volume of
    # the ball instead of concentrating them near its centre.
    r = torch.rand(*x.shape[:-1], 1, device=x.device) ** (1.0 / x.shape[-1])
    return direction * r * eps


def unit_sign(grad):
    """v = sign(g) / ||sign(g)||_2, so the step length does not depend on d."""
    sign_dir = torch.sign(grad)
    sign_norm = torch.norm(sign_dir, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return sign_dir / sign_norm


def random_attack(x_clean, eps):
    """Control: unoptimized noise with the same budget as the attacks."""
    return project_to_sphere(x_clean + l2_ball_noise(x_clean, eps), x_clean).detach()


def rfgsm_attack(loss_fn, x_clean, eps, alpha=0.5):
    """
    Randomized FGSM (Tramer et al., ICLR 2018).

    A random start within the ball of radius alpha * eps, followed by a single
    signed step whose length spends the remaining budget, treating the step as
    orthogonal to the initial displacement.
    """
    x_adv = project_to_sphere(x_clean + l2_ball_noise(x_clean, alpha * eps), x_clean).detach()
    x_adv.requires_grad_(True)
    loss_fn(x_adv).backward()
    v = unit_sign(x_adv.grad.detach())

    # The budget is measured from the real displacement after renormalization.
    base = x_adv.detach() - x_clean
    step = torch.sqrt((eps ** 2 - (base ** 2).sum(dim=-1, keepdim=True)).clamp(min=0.0))
    delta = project_l2(base + step * v, eps)
    return project_to_sphere(x_clean + delta, x_clean).detach()


def pgd_attack(loss_fn, x_clean, eps, steps=10):
    """
    Projected Gradient Descent (Madry et al., ICLR 2018).

    A random start within the ball of radius eps, followed by `steps` signed
    updates of fixed length 2.5 * eps / steps, each projected onto the L2 ball
    and renormalized onto the hypersphere.
    """
    alpha = (eps * 2.5) / steps
    delta = l2_ball_noise(x_clean, eps).detach()

    for _ in range(steps):
        x_adv = project_to_sphere(x_clean + delta, x_clean).detach()
        x_adv.requires_grad_(True)
        loss_fn(x_adv).backward()
        v = unit_sign(x_adv.grad.detach())
        delta = project_l2(x_adv.detach() + alpha * v - x_clean, eps)

    return project_to_sphere(x_clean + delta, x_clean).detach()
