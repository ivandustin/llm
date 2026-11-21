"""One versus multiple standard attention blocks under a 10k total budget."""
from experiments.benchmark import nnx,np,jax,jnp,count
from experiments.capabilities import ProbeTransformer

DEPTHS=dict(one=1,two=2,four=4)

def dimensions(task,depth):
 vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
 # Learned embeddings, final norm/head, and depth standard blocks (d16).
 # Every FF hidden unit adds 2*d+1 = 33 weights/biases per block.
 base=count(ProbeTransformer(vocab,context,16,nnx.Rngs(0),depth=depth,ff_hidden=1))
 hidden=1+(10000-base)//(33*depth)
 assert hidden>=1
 return vocab,context,hidden

class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task;self.depth=DEPTHS[architecture]
  vocab,context,hidden=dimensions(task,self.depth)
  self.base=ProbeTransformer(vocab,context,16,nnx.Rngs(seed),depth=self.depth,ff_hidden=hidden)
 def __call__(self,tokens):
  m=self.base;b,l=tokens.shape
  x=m.token(tokens)+m.position(jnp.arange(l))
  for block in m.blocks[:-1]:x=block(x)
  # Earlier blocks require all states; only final-block supervised outputs
  # need attention/FFN evaluation. Exact standard causal Transformer outputs.
  block=m.blocks[-1]
  pos=jnp.array([10] if self.task=='single' else [10,12,14] if self.task=='multi' else [l-1])
  qkv=block.qkv(block.n1(x)).reshape(b,l,3,4,4)
  q=jnp.take(qkv[:,:,0],pos,axis=1).transpose(0,2,1,3)
  k=qkv[:,:,1].transpose(0,2,1,3);v=qkv[:,:,2].transpose(0,2,1,3)
  mask=jnp.arange(l)[None,:]<=pos[:,None]
  weights=jax.nn.softmax(jnp.where(mask,(q@k.swapaxes(-1,-2))/2.,-1e9),axis=-1)
  h=jnp.take(x,pos,axis=1)+block.proj((weights@v).transpose(0,2,1,3).reshape(b,len(pos),16))
  h=h+block.down(jax.nn.gelu(block.up(block.n2(h))))
  logits=m.head(m.norm(h))
  return logits if self.task in ['single','multi'] else logits[:,0]

if __name__=='__main__':
 from pathlib import Path
 from experiments import benchmark_100k as protocol
 source=Path(__file__).with_name('benchmark_100k.py').read_text();source=source[source.index('def main():'):]
 source=source.replace("choices=['ours','transformer']","choices=['one','two','four']")
 source=source.replace("expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)","expected=count(model)")
 source=source.replace('<=100000','<=10000').replace('large-','tfblocks-')
 source=source.replace("width=64,hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1","width=16,heads=4,hidden_by_task={t:dimensions(t,DEPTHS[a.architecture])[2] for t in ['position','single','multi','language']},hidden_layers=1,blocks=DEPTHS[a.architecture],positions='learned',budget=10000")
 env=vars(protocol).copy();env.update(Model=Model,dimensions=dimensions,DEPTHS=DEPTHS,__name__='__main__');exec(compile(source,__file__,'exec'),env)
