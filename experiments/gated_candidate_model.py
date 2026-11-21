"""One-hidden-layer normalized-GELU model with a sigmoid query gate.

The query is the last input token. Its embedding is shared with the ordinary
shared token embedding table. Gating acts on pooled hidden features, not positions.
"""
from experiments.candidate_model import Model as Base
from flax import nnx
from jax import nn
import jax.numpy as jnp


class Model(Base):
    def __init__(self,embeds,classes,dims,neurons,rngs):
        super().__init__(embeds,classes,dims,neurons,rngs)
        self.query_gate=nnx.Linear(dims,neurons,rngs=rngs)

    def __call__(self,x):
        e=self.embeds(x)
        z=jnp.prod(e,axis=-3).sum(axis=-2)
        z=(z-z.mean(axis=-1,keepdims=True))/jnp.sqrt(z.var(axis=-1,keepdims=True)+1e-5)
        q=self.embeds.embeds[1](x[1,-1])
        h=nn.gelu(self.fn.a(z))*nn.sigmoid(self.query_gate(q))
        return self.fn.b(h)
