"""MLP denoiser: embeddings, network, and training step (notebook 03)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, PRNGKeyArray


def sinusoidal_embedding(t: int, dim: int = 256) -> Float[Array, " dim"]:
    """Sinusoidal time embedding (same as transformer positional encoding)."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    args = t * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


def linear(x: Float[Array, "... d_in"], layer: dict) -> Float[Array, "... d_out"]:
    """Apply a linear layer stored as {'W': ..., 'b': ...}."""
    return x @ layer["W"] + layer["b"]


def init_params(
    key: PRNGKeyArray,
    d_img: int = 3072,
    d_time: int = 256,
    d_hidden: int = 512,
) -> dict:
    """Initialise MLP denoiser parameters."""
    k0, k1, k2, k3 = jr.split(key, 4)

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
    }


def predict_noise(
    params: dict,
    xt: Float[Array, "h w c"],
    t: int,
) -> Float[Array, "h w c"]:
    """Predict the noise added to xt at timestep t."""
    t_emb = sinusoidal_embedding(t)
    t_emb = jnp.tanh(linear(t_emb, params["time_proj"]))

    x = jnp.concatenate([xt.reshape(-1), t_emb])
    x = jax.nn.gelu(linear(x, params["fc1"]))
    x = jax.nn.gelu(linear(x, params["fc2"]))
    return linear(x, params["fc3"]).reshape(xt.shape)


def train_step(
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x0_batch: Float[Array, "b h w c"],
    key: PRNGKeyArray,
    schedule: dict,
    T: int = 1000,
) -> tuple[dict, optax.OptState, Float[Array, ""]]:
    """One gradient update step on a batch of clean images."""
    from .schedule import q_sample  # avoid circular at module level

    B = x0_batch.shape[0]
    key_t, key_noise = jr.split(key)
    ts = jr.randint(key_t, (B,), 0, T)
    noise = jr.normal(key_noise, x0_batch.shape)

    def loss_fn(params):
        xt_batch = jax.vmap(q_sample, in_axes=(0, 0, 0, None))(
            x0_batch, ts, noise, schedule["alphas_bar"]
        )
        eps_pred = jax.vmap(predict_noise, in_axes=(None, 0, 0))(params, xt_batch, ts)
        return jnp.mean((noise - eps_pred) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_opt_state, loss
