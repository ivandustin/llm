"""MQAR-style causal key/value retrieval; not the official MQAR benchmark."""
import argparse,json
from pathlib import Path
import optax
from experiments.benchmark import nnx,np,jnp,jax,count
from experiments.budget_sweep import BatchedCandidate
from experiments.capabilities import ProbeTransformer
from experiments.readout_experiments import metrics,save_weights
from experiments.speed_benchmark import compiled_training


def generate(size,rng):
 keys=np.stack([rng.choice(32,5,replace=False) for _ in range(size)])
 values=rng.integers(32,48,(size,5),dtype=np.int32)
 queries=np.stack([rng.choice(5,3,replace=False) for _ in range(size)])
 x=np.empty((size,16),np.int32)
 x[:,:10:2]=keys;x[:,1:10:2]=values
 y=values[np.arange(size)[:,None],queries]
 x[:,10::2]=keys[np.arange(size)[:,None],queries]
 x[:,11::2]=y
 return x,y


class Model(nnx.Module):
 def __init__(self,kind,seed,single=False):
  self.single=single
  self.base=(BatchedCandidate(16,96,64,12,seed,6) if kind=='ours' else ProbeTransformer(96,16,16,nnx.Rngs(seed),depth=1,ff_hidden=162))
 def __call__(self,x):
  # Each query predicts its following value. Later tokens are excluded.
  return jnp.stack([self.base(x[:,:end]) for end in ([11] if self.single else [11,13,15])],axis=1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--single',action='store_true');a=p.parse_args()
 rng=np.random.default_rng(50000+a.seed)
 hx=np.empty((10000,64,16),np.int32);hy=np.empty((10000,64,3),np.int32)
 for i in range(10000):hx[i],hy[i]=generate(64,rng)
 if a.single:hy=hy[:,:,:1]
 xs,ys=jax.device_put(hx),jax.device_put(hy)
 test=generate(4096,np.random.default_rng(719001));fresh=generate(16384,np.random.default_rng(719002))
 # Verify query/answer alignment independently from generator indexing.
 for x,y in [test,fresh]:
  for q,pos in enumerate([10,12,14]):
   matches=x[:,:10:2]==x[:,pos,None]
   assert np.all(matches.sum(-1)==1)
   expected=x[:,1:10:2][np.arange(len(x)),matches.argmax(-1)]
   assert np.array_equal(expected,y[:,q]) and np.array_equal(x[:,pos+1],y[:,q])
 if a.single:
  test=(test[0],test[1][:,:1]);fresh=(fresh[0],fresh[1][:,:1])
 out=Path(f'associative-single-seed-{a.seed}.json' if a.single else f'mqar-seed-{a.seed}.json');art=out.with_suffix('');art.mkdir(exist_ok=True)
 runs=[]
 for kind in ['ours','transformer']:
  model=Model(kind,a.seed,a.single);assert count(model)==(9976 if kind=='ours' else 9970)
  state,run=compiled_training(model,optax.adam(.001))
  history=[];start=0
  for end in [1000,3000,10000]:
   executable=run.lower(state,xs[start:end],ys[start:end]).compile()
   state,losses=executable(state,xs[start:end],ys[start:end]);jax.block_until_ready((state,losses))
   nnx.update(model,state[0]);point=metrics(model,*test)|dict(step=end);history.append(point)
   print(kind,a.seed,end,point,flush=True);start=end
  # Accuracy averaged over queries plus all-three-correct sequence accuracy.
  predictions=[]
  @nnx.jit
  def pred(m,x):return m(x).argmax(-1)
  for i in range(0,len(fresh[1]),128):predictions.append(np.asarray(pred(model,fresh[0][i:i+128])))
  predicted=np.concatenate(predictions)
  result=dict(model=kind,seed=a.seed,parameters=count(model),history=history,fresh=metrics(model,*fresh),
    all_queries_correct=float((predicted==fresh[1]).all(-1).mean()))
  save_weights(model,art/f'{kind}.npz');runs.append(result)
  out.write_text(json.dumps(dict(config=dict(context=16,vocab=96,pairs=5,queries=1 if a.single else 3,batch=64,steps=10000,lr=.001,
    protocol='teacher-forced causal prefixes; same labels and data for both; no future answer access; MQAR-inspired, not official benchmark'),runs=runs),indent=2)+'\n')


if __name__=='__main__':main()
