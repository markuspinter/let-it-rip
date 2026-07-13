"""Noise schedules for the forward diffusion process."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def linear_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> dict[str, Float[Array, " T"]]:
    """Linear noise schedule from Ho et al. (2020)."""
    betas = jnp.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alphas_bar": jnp.cumprod(alphas)}


def cosine_schedule(
    T: int = 1000,
    s: float = 0.008,
) -> dict[str, Float[Array, " T"]]:
    """Cosine noise schedule from Nichol & Dhariwal (2021)."""
    steps = jnp.arange(T + 1)
    f = jnp.cos((steps / T + s) / (1 + s) * jnp.pi / 2) ** 2
    alphas_bar_full = f / f[0]
    betas = jnp.clip(1 - alphas_bar_full[1:] / alphas_bar_full[:-1], 0, 0.999)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alphas_bar": alphas_bar_full[1:]}


def q_sample(
    x0: Float[Array, "h w c"],
    t: int,
    noise: Float[Array, "h w c"],
    alphas_bar: Float[Array, " T"],
) -> Float[Array, "h w c"]:
    """Sample x_t from q(x_t | x_0) using the closed-form marginal.

    x_t = sqrt(alphas_bar[t]) * x0 + sqrt(1 - alphas_bar[t]) * noise
    """
    ab_t = alphas_bar[t]
    return jnp.sqrt(ab_t) * x0 + jnp.sqrt(1.0 - ab_t) * noise
