from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

from serl_launcher.common.common import default_init
from serl_launcher.networks.time_embedding import TimeEmbedding
from serl_launcher.diffusion.preconditioning import get_boundary_condition_scalings, append_dims
from serl_launcher.diffusion.noise_process import rescale_timesteps

class ConsistencyPolicy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    sigma_emb_dim: int
    sigma_min: float
    sigma_max: float
    rho: float
    sigma_data: float
    max_t: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, 
                    noisy_actions: jnp.ndarray, 
                    sigmas: jnp.ndarray,
                    train: bool = False):
        """
        Denoises the noisy actions using the consistency policy.
        Args:
            observations: 观测数据
            noisy_actions: 噪声动作
            sigmas: sigma噪声等级
            train: Whether to train the model.
        Returns:
            The denoised actions.
        """
        # 获取c_skip, c_out, c_in
        c_skip, c_out, c_in = [append_dims(x, noisy_actions.ndim) 
                    for x in get_boundary_condition_scalings(sigmas, self.sigma_min, self.sigma_data)]
        
        # 获取观测嵌入
        if self.encoder is None:
            obs_emb = observations
        else:
            obs_emb = self.encoder(observations, train=train, stop_gradient=True)

        # 获取时间嵌入
        t_emb = TimeEmbedding(self.sigma_emb_dim)(rescale_timesteps(sigmas, 1000.0 * 0.25))
        # 向量拼接
        inputs = jnp.concatenate([c_in * noisy_actions, t_emb, obs_emb], axis=-1)

        mlp_outputs = self.network(inputs, train=train)
        denoised = nn.Dense(self.action_dim, kernel_init=default_init())(mlp_outputs)
        denoised = c_skip * noisy_actions + c_out * denoised

        return denoised
