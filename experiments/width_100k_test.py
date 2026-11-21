"""Six hidden layers, wider learned embeddings under a 100k total budget."""
from experiments.benchmark import nnx,np,jnp,jax,count
from experiments.budget_sweep import BatchedCandidate,hidden_for_budget

WIDTHS=dict(w128=128,w256=256,w512=512)
SPECS={key:(width,hidden_for_budget(16,96,width,100000,6)) for key,width in WIDTHS.items()}
class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  width,hidden=SPECS[architecture]
  self.base=BatchedCandidate(context,vocab,width,hidden,seed,6,residual=False)
 def __call__(self,x):
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)

if __name__=='__main__':
 from pathlib import Path
 from experiments import benchmark_100k as protocol
 source=Path(__file__).with_name('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['w128','w256','w512']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('large-','width100k-')
 source=source.replace("width=64,hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","width=SPECS[a.architecture][0],hidden=SPECS[a.architecture][1],hidden_layers=6,blocks=0,residual=False,pooled_norm='non-learned epsilon1e-5',positions='learned',final_norm=False,budget=100000")
 env=vars(protocol).copy();env.update(Model=Model,SPECS=SPECS,__name__='__main__');exec(compile(source,__file__,'exec'),env)
