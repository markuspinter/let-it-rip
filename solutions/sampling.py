"""DDPM and DDIM samplers (notebook 04)."""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def ddpm_step(
    xt: Float[Array, "h w c"],
    t: int,
    eps_pred: Float[Array, "h w c"],
    schedule: dict,
    key: PRNGKeyArray,
) -> Float[Array, "h w c"]:
    """One reverse step of DDPM: x_t → x_{t-1}."""
    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alphas_bar = schedule["alphas_bar"]
    ab_t = alphas_bar[t]

    # Predict x0 and clip to the valid pixel range — without this, an
    # imperfect eps_pred gets amplified by the 1/sqrt(alphas_bar[t]) blowup
    # near t = T and the chain diverges instead of denoising.
    x0_pred = jnp.clip(
        (xt - jnp.sqrt(1.0 - ab_t) * eps_pred) / jnp.sqrt(ab_t), -1.0, 1.0
    )

    ab_prev = jnp.where(t > 0, alphas_bar[t - 1], jnp.array(1.0))
    coef_x0 = jnp.sqrt(ab_prev) * betas[t] / (1.0 - ab_t)
    coef_xt = jnp.sqrt(alphas[t]) * (1.0 - ab_prev) / (1.0 - ab_t)
    mu_t = coef_x0 * x0_pred + coef_xt * xt

    beta_tilde = (1.0 - ab_prev) / (1.0 - ab_t) * betas[t]

    z = jr.normal(key, xt.shape)
    return mu_t + jnp.where(t > 0, jnp.sqrt(beta_tilde) * z, jnp.zeros_like(xt))


def ddpm_sample(
    predict_noise_fn,
    schedule: dict,
    key: PRNGKeyArray,
    shape: tuple = (32, 32, 3),
    T: int = 1000,
) -> Float[Array, "h w c"]:
    """Generate one sample with full DDPM reverse chain (T steps)."""
    key, init_key = jr.split(key)
    xt = jr.normal(init_key, shape)
    for t in reversed(range(T)):
        key, step_key = jr.split(key)
        xt = ddpm_step(xt, t, predict_noise_fn(xt, t), schedule, step_key)
    return xt


def ddim_step(
    xt: Float[Array, "h w c"],
    t_from: int,
    t_to: int,
    eps_pred: Float[Array, "h w c"],
    schedule: dict,
) -> Float[Array, "h w c"]:
    """One deterministic DDIM step: x_{t_from} → x_{t_to}."""
    ab_from = schedule["alphas_bar"][t_from]
    ab_to = schedule["alphas_bar"][t_to]
    x0_pred = (xt - jnp.sqrt(1.0 - ab_from) * eps_pred) / jnp.sqrt(ab_from)
    x0_pred = jnp.clip(x0_pred, -1.0, 1.0)
    return jnp.sqrt(ab_to) * x0_pred + jnp.sqrt(1.0 - ab_to) * eps_pred


def ddim_sample(
    predict_noise_fn,
    schedule: dict,
    key: PRNGKeyArray,
    shape: tuple = (32, 32, 3),
    steps: int = 50,
    T: int = 1000,
) -> Float[Array, "h w c"]:
    """Generate one sample with DDIM (deterministic, fewer steps)."""
    timesteps = np.linspace(T - 1, 0, steps + 1).round().astype(int)
    xt = jr.normal(key, shape)
    for i in range(len(timesteps) - 1):
        t_from, t_to = int(timesteps[i]), int(timesteps[i + 1])
        xt = ddim_step(xt, t_from, t_to, predict_noise_fn(xt, t_from), schedule)
    return xt
