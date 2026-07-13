"""Class-conditional MLP denoiser with classifier-free guidance (notebook 05)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

NULL_CLASS = 10


def init_params(
    key: PRNGKeyArray,
    d_img: int = 3072,
    d_time: int = 256,
    d_hidden: int = 512,
    n_classes: int = 11,
) -> dict:
    """Initialise class-conditional MLP denoiser parameters."""
    k0, k1, k2, k3, k4 = jr.split(key, 5)

    def _linear(k, fan_in, fan_out):
        return {
            "W": jr.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in),
            "b": jnp.zeros(fan_out),
        }

    return {
        "time_proj": _linear(k0, d_time, d_time),
        "fc1": _linear(k1, d_img + d_time, d_hidden),
        "fc2": _linear(k2, d_hidden, d_hidden),
        "fc3": _linear(k3, d_hidden, d_img),
        "class_emb": jr.normal(k4, (n_classes, d_time)) * 0.02,
    }


def predict_noise(
    params: dict,
    xt: Float[Array, "h w c"],
    t: int,
    c: int = NULL_CLASS,
) -> Float[Array, "h w c"]:
    """Predict noise; c=NULL_CLASS for unconditional."""
    from .diffusion import sinusoidal_embedding

    t_emb = sinusoidal_embedding(t)
    t_emb = jnp.tanh(t_emb @ params["time_proj"]["W"] + params["time_proj"]["b"])
    t_emb = t_emb + params["class_emb"][c]

    x = jnp.concatenate([xt.reshape(-1), t_emb])
    x = jax.nn.gelu(x @ params["fc1"]["W"] + params["fc1"]["b"])
    x = jax.nn.gelu(x @ params["fc2"]["W"] + params["fc2"]["b"])
    return (x @ params["fc3"]["W"] + params["fc3"]["b"]).reshape(xt.shape)


def cfg_noise_estimate(
    params: dict,
    xt: Float[Array, "h w c"],
    t: int,
    c: int,
    w: float,
) -> Float[Array, "h w c"]:
    """CFG composite noise: eps_uncond + w * (eps_cond - eps_uncond)."""
    eps_uncond = predict_noise(params, xt, t, c=NULL_CLASS)
    eps_cond = predict_noise(params, xt, t, c=c)
    return eps_uncond + w * (eps_cond - eps_uncond)


def train_step(
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x0_batch: Float[Array, "b h w c"],
    labels_batch: Int[Array, " b"],
    key: PRNGKeyArray,
    schedule: dict,
    T: int = 1000,
    p_uncond: float = 0.1,
) -> tuple[dict, optax.OptState, Float[Array, ""]]:
    """One gradient update with CFG label dropout."""
    from .schedule import q_sample

    B = x0_batch.shape[0]
    key_t, key_noise, key_mask = jr.split(key, 3)
    ts = jr.randint(key_t, (B,), 0, T)
    noise = jr.normal(key_noise, x0_batch.shape)
    mask = jr.bernoulli(key_mask, p_uncond, (B,))
    labels_masked = jnp.where(mask, NULL_CLASS, labels_batch)

    def loss_fn(params):
        xt = jax.vmap(q_sample, in_axes=(0, 0, 0, None))(
            x0_batch, ts, noise, schedule["alphas_bar"]
        )
        eps_pred = jax.vmap(predict_noise, in_axes=(None, 0, 0, 0))(
            params, xt, ts, labels_masked
        )
        return jnp.mean((noise - eps_pred) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_opt_state, loss


def ddim_sample_cfg(
    params: dict,
    schedule: dict,
    key: PRNGKeyArray,
    c: int,
    w: float = 3.0,
    shape: tuple = (32, 32, 3),
    steps: int = 50,
    T: int = 1000,
) -> Float[Array, "h w c"]:
    """DDIM sampling with classifier-free guidance."""
    from .sampling import ddim_step

    timesteps = np.linspace(T - 1, 0, steps + 1).round().astype(int)
    xt = jr.normal(key, shape)
    for i in range(len(timesteps) - 1):
        t_from, t_to = int(timesteps[i]), int(timesteps[i + 1])
        xt = ddim_step(
            xt, t_from, t_to, cfg_noise_estimate(params, xt, t_from, c, w), schedule
        )
    return xt
