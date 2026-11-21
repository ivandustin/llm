"""Fixed sinusoidal positions, matched <=10k trainable parameters, compiled training."""
from experiments.benchmark import nnx,jnp,jax,count
from experiments.capabilities import Block
import math


def positions(length,width):
 p=jnp.arange(length,dtype=jnp.float32)[:,None]
 f=jnp.exp(-math.log(10000.)*jnp.arange(0,width,2,dtype=jnp.float32)/width)
 angles=p*f
 return jnp.stack([jnp.sin(angles),jnp.cos(angles)],axis=-1).reshape(length,width)


class Pool(nnx.Module):
 def __init__(self,vocab,hidden,seed):
  rng=nnx.Rngs(seed)
  self.token=nnx.Embed(vocab,64,rngs=rng)
  self.up=nnx.Linear(64,hidden,rngs=rng)
  self.layers=nnx.List([nnx.Linear(hidden,hidden,rngs=rng) for _ in range(5)])
  self.head=nnx.Linear(hidden,vocab,rngs=rng)
 def __call__(self,tokens):
  z=(self.token(tokens)*positions(tokens.shape[1],64)).sum(axis=1)
  z=(z-z.mean(-1,keepdims=True))/jnp.sqrt(z.var(-1,keepdims=True)+1e-5)
  h=jax.nn.gelu(self.up(z))
  for layer in self.layers:h=jax.nn.gelu(layer(h))
  return self.head(h)


class Transformer(nnx.Module):
 def __init__(self,vocab,hidden,seed):
  rng=nnx.Rngs(seed)
  self.token=nnx.Embed(vocab,16,rngs=rng)
  self.blocks=nnx.List([Block(16,rng,ff_hidden=hidden)])
  self.norm=nnx.LayerNorm(16,rngs=rng)
  self.head=nnx.Linear(16,vocab,rngs=rng)
 def __call__(self,tokens,task='position',full=False):
  b,l=tokens.shape
  x=self.token(tokens)+positions(l,16)
  if full:
   x=self.blocks[0](x)
   return self.head(self.norm(x))
  pos=jnp.array([10] if task=='single' else [10,12,14] if task=='multi' else [l-1])
  block=self.blocks[0]
  qkv=block.qkv(block.n1(x)).reshape(b,l,3,4,4)
  q=jnp.take(qkv[:,:,0],pos,axis=1).transpose(0,2,1,3)
  k=qkv[:,:,1].transpose(0,2,1,3);v=qkv[:,:,2].transpose(0,2,1,3)
  mask=jnp.arange(l)[None,:]<=pos[:,None]
  weights=jax.nn.softmax(jnp.where(mask,(q@k.swapaxes(-1,-2))/2.,-1e9),axis=-1)
  h=jnp.take(x,pos,axis=1)+block.proj((weights@v).transpose(0,2,1,3).reshape(b,len(pos),16))
  h=h+block.down(jax.nn.gelu(block.up(block.n2(h))))
  logits=self.head(self.norm(h))
  return logits if task in ['single','multi'] else logits[:,0]


class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task;self.architecture=architecture
  vocab=65 if task=='language' else 96
  self.base=Pool(vocab,23 if task=='language' else 15,seed) if architecture=='ours' else Transformer(vocab,201 if task=='language' else 170,seed)
 def __call__(self,x):
  if self.architecture=='transformer':return self.base(x,self.task)
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)


if __name__=='__main__':
 # Reuse exactly the previous data, update, evaluation and checkpoint protocol.
 from pathlib import Path
 source=Path(__file__).with_name('benchmark_100k.py').read_text()
 source=source[source.index('def main():'):]
 source=source.replace('expected=(94473 if task==\'language\' else 99184) if a.architecture==\'ours\' else (96665 if task==\'language\' else 99640)',"expected=(9975 if task=='language' else 9855) if a.architecture=='ours' else 9978")
 source=source.replace('<=100000','<=10000').replace('large-','sinusoidal-')
 source=source.replace('width=64,hidden=120 if a.architecture==\'ours\' else 536',"positions='fixed sinusoidal base10000',budget=10000,width=64 if a.architecture=='ours' else 16,hidden=(23 if task=='language' else 15) if a.architecture=='ours' else (201 if task=='language' else 170)")
 from experiments import benchmark_100k as protocol
 env=vars(protocol).copy();env.update(Model=Model,__name__='__main__')
 exec(compile(source,__file__,'exec'),env)
