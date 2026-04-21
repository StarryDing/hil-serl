import flax.linen as nn
import jax.numpy as jnp

from serl_launcher.diffusion.utils import sinusoidal_embedding

class TimeEmbedding(nn.Module):
    t_dim: int

    def __call__(self, t: jnp.ndarray):
        t_sinusoidal = sinusoidal_embedding(t, self.t_dim)
        t_emb = nn.Dense(self.t_dim * 2)(t_sinusoidal)
        t_emb = nn.Mish()(t_emb)
        t_emb = nn.Dense(self.t_dim)(t_emb)
        return t_emb