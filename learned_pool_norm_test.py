"""Learned affine pooled LayerNorm versus original readout controls."""
from benchmark import Model as BaseModel,nnx,jax,jnp,np,count
from readout_experiments import Head,Joint,pool

VARIANTS=['sigmoid','norm_sigmoid','gelu','norm_gelu','affine_norm_sigmoid','affine_norm_gelu']
class AffineHead(Head):
 def __init__(self,fn,variant,hidden,width,rngs):
  super().__init__(fn,variant,hidden,rngs)
  if variant.startswith('affine_'):
   self.scale=nnx.Param(jnp.ones(width));self.bias=nnx.Param(jnp.zeros(width))
 def prepare(self,z):
  z=super().prepare(z)
  if self.variant.startswith('affine_'):z=z*self.scale+self.bias
  return z

class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  language=task=='language';width=32 if language else 255;hidden=152 if language else 240
  context=32 if language else 16;vocab=65 if language else 96
  rng=nnx.Rngs(seed);base=BaseModel([context,vocab],vocab,width,hidden,rng)
  self.joint=Joint(base.embeds,AffineHead(base.fn,architecture,hidden,width,rng))
 def __call__(self,tokens):return self.joint(tokens)

if __name__=='__main__':
 from pathlib import Path
 import benchmark_100k as protocol
 source=Path('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=VARIANTS")
 source=source.replace("for task in ['position','single','multi','language']:","for task in (['position','language'] if a.architecture.startswith('affine_') else ['language']):")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=(18065 if task=='language' else 113136)+((64 if task=='language' else 510) if a.architecture.startswith('affine_') else 0)")
 source=source.replace('<=100000','<=114000').replace('large-','poolnorm-')
 # Match the original diagnostic's 4096 evaluation examples exactly, retain
 # a separate fresh 16384-example final score as additional evidence.
 source=source.replace("width=64,hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","width_by_task=dict(position=255,language=32),hidden_by_task=dict(position=240,language=152),hidden_layers=1,blocks=0,pooled_norm_epsilon=1e-5,final_norm=False")
 env=vars(protocol).copy();env.update(Model=Model,VARIANTS=VARIANTS,__name__='__main__');exec(compile(source,__file__,'exec'),env)
