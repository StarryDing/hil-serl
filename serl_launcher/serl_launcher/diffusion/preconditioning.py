import jax.numpy as jnp

def append_dims(x, target_ndim) -> jnp.ndarray:
    dims_to_append = target_ndim - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_ndim is {target_ndim}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def get_boundary_condition_scalings(sigma, sigma_min=0.002, sigma_data=0.5):
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = ((sigma - sigma_min) * sigma_data / jnp.sqrt(sigma**2 + sigma_data**2))
    c_in = 1 / jnp.sqrt(sigma**2 + sigma_data**2)
    return c_skip, c_out, c_in
