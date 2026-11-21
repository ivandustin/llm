"""Approximately 100k parameters: pooled model versus standard one-block transformer.

The transformer computes only supervised causal outputs. Logits and gradients
are checked against the original full-position implementation in
check_fast_transformer.py. All training uses compiled lax.scan.
"""
import argparse,json
from pathlib import Path
import optax
from benchmark import nnx,np,jax,jnp,count,batch
from budget_sweep import BatchedCandidate
from capabilities import ProbeTransformer,generate
from mqar_test import generate as associative
from language_depth_test import evaluate
from readout_experiments import metrics,save_weights
from speed_benchmark import compiled_training


class Model(nnx.Module):
 def __init__(self,task,seed,architecture):
  self.task=task
  self.architecture=architecture
  vocab=65 if task=='language' else 96
  context=32 if task=='language' else 16
  self.base=(BatchedCandidate(context,vocab,64,120,seed,6) if architecture=='ours' else ProbeTransformer(vocab,context,64,nnx.Rngs(seed),depth=1,ff_hidden=536))
 def __call__(self,x):
  if self.architecture=='transformer':return transformer_outputs(self.base,x,self.task)
  if self.task in ['single','multi']:
   return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.task=='single' else [11,13,15])],axis=1)
  return self.base(x)


def transformer_outputs(model,tokens,task):
 # Only requested causal positions need attention outputs and FFN work in a
 # one-block model. Keys/values still include all permitted preceding tokens.
 batch,length=tokens.shape
 positions=jnp.array([10] if task=='single' else [10,12,14] if task=='multi' else [length-1])
 x=model.token(tokens)+model.position(jnp.arange(length))
 block=model.blocks[0]
 qkv=block.qkv(block.n1(x)).reshape(batch,length,3,4,16)
 q=jnp.take(qkv[:,:,0],positions,axis=1).transpose(0,2,1,3)
 k=qkv[:,:,1].transpose(0,2,1,3);v=qkv[:,:,2].transpose(0,2,1,3)
 scores=(q@k.swapaxes(-1,-2))/4.
 mask=jnp.arange(length)[None,:]<=positions[:,None]
 weights=jax.nn.softmax(jnp.where(mask,scores,-1e9),axis=-1)
 attended=(weights@v).transpose(0,2,1,3).reshape(batch,len(positions),64)
 h=jnp.take(x,positions,axis=1)+block.proj(attended)
 h=h+block.down(jax.nn.gelu(block.up(block.n2(h))))
 logits=model.head(model.norm(h))
 return logits if task in ['single','multi'] else logits[:,0]


def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--architecture',choices=['ours','transformer'],required=True);a=p.parse_args()
 rows=[]
 chars=sorted(set(Path('data/shakespeare.txt').read_text()));enc={c:i for i,c in enumerate(chars)}
 data=np.array([enc[c] for c in Path('data/shakespeare.txt').read_text()],np.int32)
 train,val,test=np.split(data,[int(len(data)*.8),int(len(data)*.9)])
 for task in ['position','single','multi','language']:
  model=Model(task,a.seed,a.architecture)
  expected=(94473 if task=='language' else 99184) if a.architecture=='ours' else (96665 if task=='language' else 99640)
  assert count(model)==expected and count(model)<=100000
  rng=np.random.default_rng(a.seed if task=='language' else 10000*a.seed+1601 if task=='position' else 50000+a.seed)
  xs=[];ys=[]
  for _ in range(10000):
   if task=='language':x,y=batch(train,rng.integers(0,len(train)-32,size=64),32)
   elif task=='position':x,y,_=generate('indexed_copy',16,64,rng)
   else:
    x,y=associative(64,rng)
    if task=='single':y=y[:,:1]
   xs.append(x);ys.append(y)
  xs,ys=jax.device_put(np.stack(xs)),jax.device_put(np.stack(ys));jax.block_until_ready((xs,ys))
  if task=='language':
   fixed=np.random.default_rng(8675309);ev={}
   for name,part in [('validation',val),('test',test)]:
    batches=[batch(part,fixed.integers(0,len(part)-32,size=128),32) for _ in range(32)]
    ev[name]=(np.concatenate([b[0] for b in batches]),np.concatenate([b[1] for b in batches]))
   monitor=ev['validation'];final=ev['test']
  elif task=='position':
   monitor=generate('indexed_copy',16,4096,np.random.default_rng(700016))[:2]
   final=generate('indexed_copy',16,16384,np.random.default_rng(20261122))[:2]
  else:
   monitor=associative(4096,np.random.default_rng(719001));final=associative(16384,np.random.default_rng(719002))
   if task=='single':monitor=(monitor[0],monitor[1][:,:1]);final=(final[0],final[1][:,:1])
  state,run=compiled_training(model,optax.adam(.001));history=[];start=0
  for end in [1000,3000,10000]:
   executable=run.lower(state,xs[start:end],ys[start:end]).compile()
   state,losses=executable(state,xs[start:end],ys[start:end]);jax.block_until_ready((state,losses));nnx.update(model,state[0])
   point=metrics(model,*monitor)|dict(step=end);history.append(point);print(a.architecture,task,a.seed,end,point,flush=True);start=end
  result=dict(task=task,seed=a.seed,parameters=count(model),history=history,final=evaluate(model,*final))
  directory=Path(f'large-{a.architecture}-seed-{a.seed}');directory.mkdir(exist_ok=True)
  save_weights(model,directory/f'{task}.npz');rows.append(result)
  Path(f'large-{a.architecture}-seed-{a.seed}.json').write_text(json.dumps(dict(config=dict(architecture=a.architecture,width=64,hidden=120 if a.architecture=='ours' else 536,hidden_layers=6 if a.architecture=='ours' else 1,blocks=0 if a.architecture=='ours' else 1,steps=10000,batch=64,lr=.001),runs=rows),indent=2)+'\n')


if __name__=='__main__':main()
