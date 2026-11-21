"""Only pooled models: residual additions at 6/8/12 hidden layers, <=100k."""
from benchmark import nnx,np,jnp,jax,count
from budget_sweep import BatchedCandidate

SPECS=dict(six=(6,120),eight=(8,103),twelve=(12,84))
class Model(nnx.Module):
 def __init__(self,task,seed,architecture,residual=True):
  self.task=task
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  layers,hidden=SPECS[architecture]
  self.base=BatchedCandidate(context,vocab,64,hidden,seed,layers,residual=residual)
 def __call__(self,x):
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)

if __name__=='__main__':
 from pathlib import Path
 import benchmark_100k as protocol
 source=Path('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['six','eight','twelve']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('large-','resdepth100k-')
 source=source.replace("hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","hidden=SPECS[a.architecture][1],hidden_layers=SPECS[a.architecture][0],blocks=0,residual=True,residual_rule='h + GELU(W h + b) on equal-width hidden transitions only',residual_prenorm=False,pooled_norm='non-learned epsilon1e-5',positions='learned',final_norm=False,budget=100000")
 env=vars(protocol).copy();env.update(Model=Model,SPECS=SPECS,__name__='__main__');exec(compile(source,__file__,'exec'),env)
