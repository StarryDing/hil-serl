from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

from serl_launcher.networks.time_embedding import TimeEmbedding

class ConsistencyPolicy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    t_emb_dim: int = 16
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    max_t: int = 40

    def sample_sigma(self, t):
        pass

    def __call__(self, observations: jnp.ndarray, 
                    noisy_actions: jnp.ndarray, 
                    t: jnp.ndarray,
                    train: bool = False):
        if self.encoder is None:
            obs_emb = observations
        else:
            obs_emb = self.encoder(observations, train=train, stop_gradient=True)

        t_emb = TimeEmbedding(self.t_emb_dim)(t)
        
        outputs = self.network(obs_emb, train=train)
