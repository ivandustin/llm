"""Two-hidden-layer normalized-GELU variant of the pooling model."""
from candidate_model import Model as Shallow
from flax import nnx
from jax import nn
import jax.numpy as jnp


class Model(Shallow):
    def __init__(self,embeds,classes,dims,neurons,rngs):
        super().__init__(embeds,classes,dims,neurons,rngs)
        self.extra=nnx.Linear(neurons,neurons,rngs=rngs)

    def __call__(self,x):
        e=self.embeds(x)
        z=jnp.prod(e,axis=-3).sum(axis=-2)
        z=(z-z.mean(axis=-1,keepdims=True))/jnp.sqrt(z.var(axis=-1,keepdims=True)+1e-5)
        h=nn.gelu(self.fn.a(z))
        return self.fn.b(nn.gelu(self.extra(h)))
