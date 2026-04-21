import math
import jax.numpy as jnp

def sinusoidal_embedding(t: jnp.ndarray, dim: int):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.cat((jnp.sin(emb), jnp.cos(emb)), dim=-1)
    return emb 