import jax
import jax.numpy as jnp


def khatri_rao(A, B):
    "Compute the Khatri-Rao product. Unavailable in jax.numpy for some reason."
    compute_cols = jax.vmap(lambda a, b: jnp.outer(b, a).ravel(), in_axes=1, out_axes=1)
    return compute_cols(A, B)
