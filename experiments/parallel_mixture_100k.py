"""Shared d64 pooling, parallel single-hidden FFNs, equal probability mixture."""
import math
from experiments.benchmark import nnx,np,jnp,jax,count

BRANCHES=dict(n1=1,n2=2,n4=4,n8=8)
def hidden_for_branches(n):
 # Query budget: one shared position/token table, independent biased FFNs.
 return (100000-(16+96)*64-n*96)//(n*(64+96+1))

class Branch(nnx.Module):
 def __init__(self,hidden,vocab,rngs):
  self.up=nnx.Linear(64,hidden,rngs=rngs)
  self.down=nnx.Linear(hidden,vocab,rngs=rngs)
 def __call__(self,z):return self.down(jax.nn.gelu(self.up(z)))

class Pool(nnx.Module):
 def __init__(self,context,vocab,n,seed):
  rng=nnx.Rngs(seed);hidden=hidden_for_branches(n)
  self.position=nnx.Embed(context,64,rngs=rng)
  self.token=nnx.Embed(vocab,64,rngs=rng)
  self.branches=nnx.List([Branch(hidden,vocab,rng) for _ in range(n)])
 def __call__(self,tokens):
  z=(self.token(tokens)*self.position(jnp.arange(tokens.shape[1]))).sum(axis=1)
  z=(z-z.mean(-1,keepdims=True))/jnp.sqrt(z.var(-1,keepdims=True)+1e-5)
  logps=jnp.stack([jax.nn.log_softmax(branch(z),axis=-1) for branch in self.branches])
  # Returning normalized log probabilities preserves the common CE interface:
  # log_softmax(log p) = log p. Never average logits here.
  return jax.scipy.special.logsumexp(logps,axis=0)-math.log(len(self.branches))

class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task
  self.base=Pool(32 if task=='language' else 16,65 if task=='language' else 96,BRANCHES[architecture],seed)
 def __call__(self,x):
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)

if __name__=='__main__':
 from pathlib import Path
 from experiments import benchmark_100k as protocol
 source=Path(__file__).with_name('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['n1','n2','n4','n8']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('large-','mixture100k-')
 source=source.replace("width=64,hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","width=64,hidden=hidden_for_branches(BRANCHES[a.architecture]),hidden_layers=1,branches=BRANCHES[a.architecture],branch_weights='equal1/N, average probabilities after branch softmax',blocks=0,residual=False,pooled_norm='non-learned epsilon1e-5',positions='learned',final_norm=False,budget=100000")
 env=vars(protocol).copy();env.update(Model=Model,BRANCHES=BRANCHES,hidden_for_branches=hidden_for_branches,__name__='__main__');exec(compile(source,__file__,'exec'),env)
