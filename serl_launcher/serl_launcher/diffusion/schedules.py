import jax.numpy as jnp

def karras_sigmas(sigma_min, sigma_max, rho, steps, append_zero=False) -> jnp.ndarray:
    """
    Karras sigmas for consistency training.
    """
    if steps <= 0:
        raise ValueError(f"steps must be greater than 0, got {steps}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be > 0")
    if sigma_min > sigma_max:
        raise ValueError("sigma_min must be <= sigma_max")
    if rho <= 0:
        raise ValueError("rho must be > 0")

    ramp = jnp.linspace(0, 1.0, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    if append_zero:
        sigmas = jnp.concatenate([sigmas, jnp.zeros(1, dtype=sigmas.dtype)], axis=0)

    return sigmas

