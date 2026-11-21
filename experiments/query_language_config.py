"""Test the language-selected 16d/30-wide deep model on queried copying."""
import argparse,json
from pathlib import Path
import optax
from experiments.benchmark import nnx,np,jnp,jax,count
from experiments.budget_sweep import BatchedCandidate
from experiments.capabilities import generate
from experiments.readout_experiments import metrics,save_weights
from experiments.speed_benchmark import compiled_training


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--seed',type=int,default=0)
 p.add_argument('--width',type=int,default=16)
 p.add_argument('--hidden',type=int,default=30)
 p.add_argument('--layers',type=int,default=6)
 p.add_argument('--output',default='query-language-seed-0.json')
 a=p.parse_args()
 context,vocab,steps=16,96,10000
 model=BatchedCandidate(context,vocab,a.width,a.hidden,a.seed,a.layers)
 params=count(model)
 assert params<=10000
 rng=np.random.default_rng(10000*a.seed+1601)
 hx=np.empty((steps,64,context),np.int32);hy=np.empty((steps,64),np.int32)
 for i in range(steps):hx[i],hy[i],_=generate('indexed_copy',context,64,rng)
 xs,ys=jax.device_put(hx),jax.device_put(hy)
 jax.block_until_ready((xs,ys))
 test=generate('indexed_copy',context,4096,np.random.default_rng(700016))[:2]
 fresh=generate('indexed_copy',context,16384,np.random.default_rng(20261122))[:2]
 state,run=compiled_training(model,optax.adam(.001))
 history=[];begin=0
 for end in [1000,3000,10000]:
  executable=run.lower(state,xs[begin:end],ys[begin:end]).compile()
  state,losses=executable(state,xs[begin:end],ys[begin:end])
  jax.block_until_ready((state,losses));nnx.update(model,state[0])
  point=metrics(model,*test)|dict(step=end)
  history.append(point);print(a.seed,end,point['accuracy'],flush=True);begin=end
 out=Path(a.output);artifacts=out.with_suffix('');artifacts.mkdir(exist_ok=True)
 save_weights(model,artifacts/f'model-seed-{a.seed}.npz')
 result=dict(seed=a.seed,width=a.width,hidden=a.hidden,hidden_layers=a.layers,parameters=params,steps=steps,
  history=history,fresh=metrics(model,*fresh),evaluation_seed=700016,fresh_seed=20261122)
 out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
