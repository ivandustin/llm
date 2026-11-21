"""Paired ablation: learned final LayerNorm with learned position embeddings."""
from benchmark import nnx,jax,jnp,count,np
from budget_sweep import BatchedCandidate
from stacked_candidate_model import Model as Stacked

class FinalNormPool(Stacked):
 def __init__(self,context,vocab,seed,enabled,learned=True):
  super().__init__([context,vocab],vocab,64,12,nnx.Rngs(seed),hidden_layers=6)
  self.enabled=enabled
  if enabled:self.final_norm=nnx.LayerNorm(12,epsilon=1e-6,use_scale=learned,use_bias=learned,rngs=nnx.Rngs(seed+1000))
 def __call__(self,x):
  e=self.embeds(x)
  z=jnp.prod(e,axis=-3).sum(axis=-2)
  z=(z-z.mean(axis=-1,keepdims=True))/jnp.sqrt(z.var(axis=-1,keepdims=True)+1e-5)
  h=jax.nn.gelu(self.fn.a(z))
  for layer in self.layers:h=jax.nn.gelu(layer(h))
  if self.enabled:h=self.final_norm(h)
  return self.fn.b(h)

class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  self.base=BatchedCandidate(context,vocab,64,12,seed,6)
  self.base.model=FinalNormPool(context,vocab,seed,architecture!='plain',learned=architecture=='final')
 def __call__(self,x):
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)

if __name__=='__main__':
 from pathlib import Path
 import benchmark_100k as protocol
 source=Path('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['plain','final','fixed']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=(8613 if task=='language' else 9976)+(24 if a.architecture=='final' else 0)")
 source=source.replace('<=100000','<=10000').replace('large-','finalnorm-')
 source=source.replace("hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","hidden=12,hidden_layers=6,blocks=0,final_norm=a.architecture!='plain',final_norm_learned=a.architecture=='final',final_norm_epsilon=1e-6,positions='learned',pooled_norm='unchanged parameter-free'")
 env=vars(protocol).copy();env.update(Model=Model,__name__='__main__');exec(compile(source,__file__,'exec'),env)
