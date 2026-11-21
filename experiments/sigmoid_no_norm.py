"""Original sigmoid/no-normalization pipeline, paired with the wide shallow control."""
from experiments.benchmark import jax,jnp
from experiments.parallel_mixture_100k import Pool as SharedPool,Model as SharedModel


class Pool(SharedPool):
    def __call__(self,tokens):
        z=(self.token(tokens)*self.position(jnp.arange(tokens.shape[1]))).sum(axis=1)
        branch=self.branches[0]
        return branch.down(jax.nn.sigmoid(branch.up(z)))


class Model(SharedModel):
    def __init__(self,task,seed):
        self.task=task
        self.base=Pool(32 if task=='language' else 16,
                       65 if task=='language' else 96,1,seed)
