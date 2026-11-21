"""Configurable-depth normalized-GELU readout for the pooling model.

Hidden-layer count includes the original input-to-hidden affine layer.
All added hidden layers have equal width and GELU activations. Optional
residual connections add the previous hidden vector without adding parameters.
"""
from .normalized import Model as Base
from flax import nnx
from jax import nn
import jax.numpy as jnp


class Model(Base):
    def __init__(self,embeds,classes,dims,neurons,rngs,hidden_layers=3,residual=False,residual_prenorm=False):
        if hidden_layers<1:raise ValueError('hidden_layers must be positive')
        super().__init__(embeds,classes,dims,neurons,rngs)
        if residual_prenorm and not residual:raise ValueError("prenorm requires residual connections")
        self.residual_prenorm=residual_prenorm
        self.residual=residual
        self.layers=nnx.List([nnx.Linear(neurons,neurons,rngs=rngs) for _ in range(hidden_layers-1)])

    def __call__(self,x):
        e=self.embeds(x)
        z=jnp.prod(e,axis=-3).sum(axis=-2)
        z=(z-z.mean(axis=-1,keepdims=True))/jnp.sqrt(z.var(axis=-1,keepdims=True)+1e-5)
        h=nn.gelu(self.fn.a(z))
        for layer in self.layers:
            branch=h
            if self.residual_prenorm:
                branch=(h-h.mean(axis=-1,keepdims=True))/jnp.sqrt(h.var(axis=-1,keepdims=True)+1e-5)
            delta=nn.gelu(layer(branch))
            h=h+delta if self.residual else delta
        return self.fn.b(h)
