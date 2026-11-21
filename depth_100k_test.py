"""Deeper fixed-budget pooling and a two-block Transformer reference."""
from benchmark import nnx,np,jnp,jax,count
from budget_sweep import BatchedCandidate
from capabilities import ProbeTransformer
from two_block_100k import Model as TwoBlock

SPECS=dict(eight=(8,103),twelve=(12,84))
class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task;self.architecture=architecture
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  if architecture=='transformer':
   self.base=TwoBlock(task,seed)
   # Keep identical internal dimensions across language and retrieval.
   self.base.hidden=200
   self.base.base=ProbeTransformer(vocab,context,64,nnx.Rngs(seed),depth=2,ff_hidden=200)
  else:
   layers,hidden=SPECS[architecture]
   self.base=BatchedCandidate(context,vocab,64,hidden,seed,layers)
 def __call__(self,x):
  if self.architecture=='transformer':return self.base(x)
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)

if __name__=='__main__':
 from pathlib import Path
 import benchmark_100k as protocol
 source=Path('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['eight','twelve','transformer']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('large-','depth100k-')
 source=source.replace("hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","hidden=200 if a.architecture=='transformer' else SPECS[a.architecture][1],hidden_layers=1 if a.architecture=='transformer' else SPECS[a.architecture][0],blocks=2 if a.architecture=='transformer' else 0,pooled_norm='non-learned',positions='learned',residual=a.architecture=='transformer',final_norm=a.architecture=='transformer',budget=100000")
 env=vars(protocol).copy();env.update(Model=Model,SPECS=SPECS,__name__='__main__');exec(compile(source,__file__,'exec'),env)
