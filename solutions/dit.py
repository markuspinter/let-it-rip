"""Diffusion Transformer (DiT) with AdaLN and CFG (notebook 06)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
from jaxtyping import Array, Float, Int

PATCH_SIZE = 4
IMAGE_SIZE = 32
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
D_MODEL = 256
N_HEADS = 4
DEPTH = 6
NULL_CLASS = 10


def sinusoidal_embedding(t: int, dim: int = D_MODEL) -> Float[Array, " dim"]:
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    args = t * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


class TimestepEmbedder(nnx.Module):
    def __init__(self, d: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(d, d, rngs=rngs)
        self.d = d

    def __call__(self, t: Int[Array, " b"]) -> Float[Array, "b d"]:
        embs = jax.vmap(lambda ti: sinusoidal_embedding(ti, self.d))(t)
        return jnp.tanh(self.proj(embs))


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size: int, in_ch: int, d: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.proj = nnx.Linear(patch_size * patch_size * in_ch, d, rngs=rngs)

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b n d"]:
        B, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, -1, p * p * C)
        return self.proj(x)


class PatchUnembed(nnx.Module):
    def __init__(
        self, patch_size: int, out_ch: int, d: int, n_per_side: int, rngs: nnx.Rngs
    ):
        self.patch_size = patch_size
        self.n_per_side = n_per_side
        self.proj = nnx.Linear(d, patch_size * patch_size * out_ch, rngs=rngs)

    def __call__(self, x: Float[Array, "b n d"]) -> Float[Array, "b h w c"]:
        B, p, G = x.shape[0], self.patch_size, self.n_per_side
        x = self.proj(x)
        x = x.reshape(B, G, G, p, p, -1)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, G * p, G * p, -1)


class SelfAttention(nnx.Module):
    def __init__(self, d: int, n_heads: int, rngs: nnx.Rngs):
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nnx.Linear(d, 3 * d, rngs=rngs)
        self.out = nnx.Linear(d, d, rngs=rngs)

    def __call__(self, x: Float[Array, "b n d"]) -> Float[Array, "b n d"]:
        B, N, d = x.shape
        H, Dh = self.n_heads, self.head_dim
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(Dh), axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, d)
        return self.out(out)


class AdaLNBlock(nnx.Module):
    def __init__(self, d: int, n_heads: int, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(d, use_bias=False, use_scale=False, rngs=rngs)
        self.attn = SelfAttention(d, n_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d, use_bias=False, use_scale=False, rngs=rngs)
        self.mlp1 = nnx.Linear(d, 4 * d, rngs=rngs)
        self.mlp2 = nnx.Linear(4 * d, d, rngs=rngs)
        self.adaLN = nnx.Linear(d, 4 * d, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "b n d"],
        cond: Float[Array, "b d"],
    ) -> Float[Array, "b n d"]:
        shift1, scale1, shift2, scale2 = jnp.split(self.adaLN(cond), 4, axis=-1)
        shift1, scale1 = shift1[:, None], scale1[:, None]
        shift2, scale2 = shift2[:, None], scale2[:, None]
        x = x + self.attn(self.norm1(x) * (1 + scale1) + shift1)
        x = x + self.mlp2(jax.nn.gelu(self.mlp1(self.norm2(x) * (1 + scale2) + shift2)))
        return x


class DiT(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.patch_embed = PatchEmbed(PATCH_SIZE, 3, D_MODEL, rngs=rngs)
        self.patch_unembed = PatchUnembed(
            PATCH_SIZE, 3, D_MODEL, IMAGE_SIZE // PATCH_SIZE, rngs=rngs
        )
        self.pos_embed = nnx.Param(jnp.zeros((N_PATCHES, D_MODEL)))
        self.class_embed = nnx.Param(
            jr.normal(rngs.params(), (NULL_CLASS + 1, D_MODEL)) * 0.02
        )
        self.time_embed = TimestepEmbedder(D_MODEL, rngs=rngs)
        self.blocks = [AdaLNBlock(D_MODEL, N_HEADS, rngs=rngs) for _ in range(DEPTH)]
        self.final_norm = nnx.LayerNorm(D_MODEL, rngs=rngs)

    def __call__(
        self,
        xt: Float[Array, "b h w c"],
        t: Int[Array, " b"],
        c: Int[Array, " b"],
    ) -> Float[Array, "b h w c"]:
        x = self.patch_embed(xt) + self.pos_embed.value[None]
        cond = self.time_embed(t) + self.class_embed.value[c]
        for block in self.blocks:
            x = block(x, cond)
        return self.patch_unembed(self.final_norm(x))
