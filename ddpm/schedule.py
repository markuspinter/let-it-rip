"""Noise schedules for the forward diffusion process."""

from jaxtyping import Array, Float


def linear_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> dict[str, Float[Array, " T"]]:
    """Linear noise schedule from Ho et al. (2020)."""
    # YOUR CODE HERE
    raise NotImplementedError


def cosine_schedule(
    T: int = 1000,
    s: float = 0.008,
) -> dict[str, Float[Array, " T"]]:
    """Cosine noise schedule from Nichol & Dhariwal (2021)."""
    # YOUR CODE HERE
    raise NotImplementedError


def q_sample(
    x0: Float[Array, "h w"],
    t: int,
    noise: Float[Array, "h w"],
    alphas_bar: Float[Array, " T"],
) -> Float[Array, "h w"]:
    """Sample x_t from q(x_t | x_0) using the closed-form marginal.

    x_t = sqrt(alphas_bar[t]) * x0 + sqrt(1 - alphas_bar[t]) * noise
    """
    # YOUR CODE HERE
    raise NotImplementedError
