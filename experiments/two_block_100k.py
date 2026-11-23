"""Two-block standard Transformer at the prior ~100k query budget."""
from experiments.benchmark import nnx,jax,jnp,count,np
from experiments.capabilities import ProbeTransformer
from experiments.benchmark_100k import transformer_outputs

class Model(nnx.Module):
 def __init__(self,task,seed,architecture='transformer'):
  self.task=task
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  target=94473 if task=='language' else 99184
  base=count(ProbeTransformer(vocab,context,64,nnx.Rngs(seed),depth=2,ff_hidden=1))
  self.hidden=1+(target-base)//258
  self.base=ProbeTransformer(vocab,context,64,nnx.Rngs(seed),depth=2,ff_hidden=self.hidden)
 def __call__(self,tokens):
  m=self.base;b,l=tokens.shape
  width=m.token.embedding.shape[-1];head_width=width//4
  x=m.token(tokens)+m.position(jnp.arange(l))
  x=m.blocks[0](x)
  block=m.blocks[1]
  pos=jnp.array([10] if self.task=='single' else [10,12,14] if self.task=='multi' else [l-1])
  qkv=block.qkv(block.n1(x)).reshape(b,l,3,4,head_width)
  q=jnp.take(qkv[:,:,0],pos,axis=1).transpose(0,2,1,3)
  k=qkv[:,:,1].transpose(0,2,1,3);v=qkv[:,:,2].transpose(0,2,1,3)
  mask=jnp.arange(l)[None,:]<=pos[:,None]
  w=jax.nn.softmax(jnp.where(mask,(q@k.swapaxes(-1,-2))/jnp.sqrt(float(head_width)),-1e9),axis=-1)
  h=jnp.take(x,pos,axis=1)+block.proj((w@v).transpose(0,2,1,3).reshape(b,len(pos),width))
  h=h+block.down(jax.nn.gelu(block.up(block.n2(h))))
  logits=m.head(m.norm(h))
  return logits if self.task in ['single','multi'] else logits[:,0]

if __name__=='__main__':
 from pathlib import Path
 from experiments import benchmark_100k as protocol
 source=Path(__file__).with_name('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("for task in ['position','single','multi','language']:","for task in ['position']:")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('large-','two100k-')
 source=source.replace("hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","hidden=model.hidden,hidden_layers=1,blocks=2")
 env=vars(protocol).copy();env.update(Model=Model,__name__='__main__');exec(compile(source,__file__,'exec'),env)
