"""Parameter-free sinusoidal positions, added to Transformer token embeddings."""
import math
from flax import nnx
import jax.numpy as jnp


class SinusoidalPositions(nnx.Module):
    def __init__(self, width):
        self.width = width

    def __call__(self, indices):
        frequencies = jnp.exp(-math.log(10000.) *
                              jnp.arange(0, self.width, 2, dtype=jnp.float32) / self.width)
        angles = jnp.asarray(indices, dtype=jnp.float32)[..., None] * frequencies
        return jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(
            (*indices.shape, -1))[..., :self.width]
