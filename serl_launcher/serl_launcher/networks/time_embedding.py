import math
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable

from serl_launcher.common.common import default_init

def sinusoidal_embedding(t: jnp.ndarray, dim: int):
    """
    Sinusoidal embedding.
    Args:
        t: The time steps.
        dim: The dimension of the embedding.
    Returns:
        The sinusoidal embedding.
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
    return emb 

class TimeEmbedding(nn.Module):
    """
    Time embedding module.
    Args:
        t_dim: The dimension of the time embedding.
    """
    t_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.swish

    @nn.compact
    def __call__(self, t: jnp.ndarray):
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        t_sinusoidal = sinusoidal_embedding(t, self.t_dim)
        t_emb = nn.Dense(self.t_dim * 2, kernel_init=default_init())(t_sinusoidal)
        t_emb = activations(t_emb)
        t_emb = nn.Dense(self.t_dim, kernel_init=default_init())(t_emb)
        return t_emb
        