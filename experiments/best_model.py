"""Retained architecture selected by the queried-copy experiments.

Six GELU hidden layers, width 12, embedding dimension 64; pooled normalization;
no query gate and no residuals. Token/position table sizes depend on the task.
This is the best retrieval architecture tested, not a claim of universal optimality.
"""
from experiments.stacked_candidate_model import Model
from experiments.benchmark import nnx,jnp

ARCHITECTURE=dict(width=64,hidden=12,hidden_layers=6,residual=False)


def create(context,vocab,seed=0):
    return Model([context,vocab],vocab,64,12,nnx.Rngs(seed),hidden_layers=6,residual=False)


class Batched(nnx.Module):
    def __init__(self,context,vocab,seed=0):
        self.model=create(context,vocab,seed)

    def __call__(self,tokens):
        positions=jnp.broadcast_to(jnp.arange(tokens.shape[-1]),tokens.shape)
        return nnx.vmap(self.model)(jnp.stack([positions,tokens],axis=1))
