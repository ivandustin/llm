"""Learned position-token product pooling and the original sigmoid readout.

An example has shape (fields, length), with position indices first and token
indices second. Each field has its own embedding table. Products across fields
are summed across sequence positions before the feed-forward readout.
"""
from flax import nnx
import jax.numpy as jnp


class Embeds(nnx.Module):
    """One learned embedding table per discrete input field."""

    def __init__(self, width: int, lengths: list[int], rngs: nnx.Rngs):
        self.embeds = nnx.List([nnx.Embed(n, width, rngs=rngs) for n in lengths])

    def __call__(self, xs):
        return jnp.array([self.embeds[i](x) for i, x in enumerate(xs)])


class Readout(nnx.Module):
    """Product/sum pooling followed by a single sigmoid hidden layer."""

    def __init__(self, din: int, dout: int, hidden: int, rngs: nnx.Rngs):
        self.a = nnx.Linear(din, hidden, rngs=rngs)
        self.b = nnx.Linear(hidden, dout, rngs=rngs)

    def __call__(self, embeddings):
        pooled = jnp.prod(embeddings, axis=-3).sum(axis=-2)
        return self.b(nnx.sigmoid(self.a(pooled)))


class Model(nnx.Module):
    """Original pooling model; normalized GELU variants inherit its parameters.

    ``fn`` and ``embeds`` names and readout-before-embedding initialization are
    retained for compatibility with the repository's saved parameter files.
    """

    def __init__(self, embeds: list[int], classes: int, dims: int,
                 neurons: int, rngs: nnx.Rngs):
        self.fn = Readout(dims, classes, neurons, rngs=rngs)
        self.embeds = Embeds(dims, embeds, rngs=rngs)

    def __call__(self, x):
        return self.fn(self.embeds(x))
