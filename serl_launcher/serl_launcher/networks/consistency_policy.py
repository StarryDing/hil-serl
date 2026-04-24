from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

from serl_launcher.common.common import default_init
from serl_launcher.diffusion.preconditioning import get_boundary_condition_scalings, append_dims
from serl_launcher.diffusion.noise_process import rescale_timesteps

class ConsistencyPolicy(nn.Module):
    encoder: Optional[nn.Module]
    train_encoder: bool
    network: nn.Module
    t_network: nn.Module
    action_dim: int
    sigma_min: float
    sigma_max: float
    rho: float
    sigma_data: float
    clip_denoised: bool

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
            if self.train_encoder:
                obs_emb = self.encoder(observations, train=True)
            else:
                obs_emb = self.encoder(observations, train=False, stop_gradient=True)

        # 获取时间嵌入
        rescale_t = rescale_timesteps(sigmas, 1000.0 * 0.25)
        t_emb = self.t_network(rescale_t)
        # 向量拼接
        inputs = jnp.concatenate([c_in * noisy_actions, t_emb, obs_emb], axis=-1)

        mlp_outputs = self.network(inputs, train=train)
        denoised = nn.Dense(self.action_dim, kernel_init=default_init())(mlp_outputs)
        denoised = c_skip * noisy_actions + c_out * denoised
        if self.clip_denoised:
            denoised = jnp.clip(denoised, -1, 1)

        return denoised
