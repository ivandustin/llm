"""Experimental normalized-GELU readout for the position × token model.

Preserves the original learned embeddings, elementwise product and sum.
Adds parameter-free layer normalization before the original affine layers,
and changes the hidden activation from sigmoid to GELU. Inherits the base model's
constructor and per-example position/token-index input convention.
"""
from .base import Model as Base
from jax import nn
import jax.numpy as jnp


class Model(Base):
    def __call__(self,x):
        e=self.embeds(x)
        z=jnp.prod(e,axis=-3).sum(axis=-2)
        z=(z-z.mean(axis=-1,keepdims=True))/jnp.sqrt(z.var(axis=-1,keepdims=True)+1e-5)
        return self.fn.b(nn.gelu(self.fn.a(z)))
