from typing import Optional
import jax.numpy as jnp
import jax

from serl_launcher.diffusion.preconditioning import append_dims

def rescale_timesteps(sigmas: jnp.ndarray, scale_factor: float = 1000.0 * 0.25) -> jnp.ndarray:
    """
    将 sigma 重新缩放为连续时间坐标，以便进行正弦嵌入。
        1. 由于 sigma 跨越多个数量级，因此使用 log(sigma)。
        2. 在对数空间中，sigma 的乘法变化会转化为加法偏移，
            这使得在不同噪声水平下的条件数更加平滑。
        3. 乘以一个缩放因子，使得所得的“时间”量级
    适合正弦波时间步长嵌入（大致相当于经典扩散算法中使用的 0~1000 时间步长范围）。
    """
    return scale_factor * jnp.log(sigmas + 1e-44)

def sample_sigmas(rng: jax.Array, sigma_schedule: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """
    Samples sigmas from the sigma schedule.
    Args:
        rng: The random number generator.
        sigma_schedule: The sigma schedule.
        batch_size: The batch size.
    Returns:
        The sampled sigmas.
    """
    indices = jax.random.randint(
            rng, 
            shape=(batch_size,), 
            minval=0, 
            maxval=sigma_schedule.shape[0]
        )
    sigmas = sigma_schedule[indices]
    return sigmas

def add_ve_noise(clean_actions: jnp.ndarray, sigmas: jnp.ndarray, rng: Optional[jax.Array]) -> jnp.ndarray:
    """
    Adds Variance Exploding noise to the actions.
    Args:
        clean_actions: The clean actions.
            shape: (batch_size, action_dim)
        sigmas: The sigmas.
            shape: (batch_size,)
        rng: The random number generator.
    Returns:
        The actions with variance exploration noise and the noise.
    """
    if clean_actions.shape[0] != sigmas.shape[0]:
        raise ValueError(f"Actions and sigmas must have the same number of dimensions, got {clean_actions.shape[0]} and {sigmas.shape[0]}")
    noise = jax.random.normal(rng, shape=clean_actions.shape)
    sigmas = append_dims(sigmas, clean_actions.ndim)
    return (clean_actions + noise * sigmas), noise

def make_init_noisy_action_ve(rng, sigma_max, action_dim, batch_size):
    """
    Makes initial noise actions for the actions.
    Args:
        rng: The random number generator.
        sigma_max: The maximum sigma.
        action_dim: The dimension of the actions.
        batch_size: The batch size.
    Returns:
        The initial noise actions.
    """
    noise = jax.random.normal(rng, shape=(batch_size, action_dim))
    sigmas = jnp.ones((batch_size, )) * sigma_max
    noisy_actions = noise * sigmas
    return noisy_actions, sigmas
    
def add_vp_noise(clean_actions: jnp.ndarray, sigmas: jnp.ndarray, rng: Optional[jax.Array]) -> jnp.ndarray:
    """
    Adds Variance Preserving noise to the actions.
    Args:
        actions: The actions.
        sigmas: The sigmas.
    Returns:
        The actions with variance preserving noise.
    """
    pass