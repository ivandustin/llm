"""Chunked standard-task training, with 10k parity checks for existing models.

The token_product variant tests last-token-conditioned pooling without adding
parameters, using the same data and evaluation protocol as the other models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import optax
from experiments.benchmark import nnx,np,jax,jnp,count,batch
from experiments.benchmark_100k import Model as DeepModel
from experiments.parallel_mixture_100k import Model as ShallowModel
from experiments.sigmoid_no_norm import Model as SigmoidModel
from experiments.depth_100k_test import Model as TransformerModel
from experiments.capabilities import generate
from experiments.mqar_test import generate as associative
from experiments.language_depth_test import evaluate
from experiments.readout_experiments import metrics,save_weights
from experiments.speed_benchmark import compiled_training
from experiments.last_token_probe import TokenInteraction


class TokenProductModel(nnx.Module):
    """Shared last-token multiplication for every task; no task-specific retrieval."""
    def __init__(self, task, seed):
        self.task = task
        self.base = TokenInteraction(seed, 'token_product', 576, 1,
                                    32 if task == 'language' else 16,
                                    65 if task == 'language' else 96)

    def __call__(self, x):
        if self.task in ['single', 'multi']:
            ends = [11] if self.task == 'single' else [11, 13, 15]
            return jnp.stack([self.base(x[:, :end]) for end in ends], axis=1)
        return self.base(x)

TASKS=['position','single','multi','language']

def training_batch(task,rng,train):
    if task=='language':return batch(train,rng.integers(0,len(train)-32,size=64),32)
    if task=='position':return generate('indexed_copy',16,64,rng)[:2]
    x,y=associative(64,rng)
    return x,y[:,:1] if task=='single' else y


def evaluation_data(task,val,test):
    if task=='language':
        fixed=np.random.default_rng(8675309);ev=[]
        for part in [val,test]:
            batches=[batch(part,fixed.integers(0,len(part)-32,size=128),32) for _ in range(32)]
            ev.append((np.concatenate([b[0] for b in batches]),np.concatenate([b[1] for b in batches])))
        return ev
    if task=='position':
        return [generate('indexed_copy',16,n,np.random.default_rng(seed))[:2]
                for n,seed in [(4096,700016),(16384,20261122)]]
    ev=[associative(n,np.random.default_rng(seed)) for n,seed in [(4096,719001),(16384,719002)]]
    return [(x,y[:,:1]) for x,y in ev] if task=='single' else ev


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--architecture',choices=['deep','shallow','sigmoid','transformer','token_product'],default='deep')
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--chunk',type=int,default=1000)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    args=parser.parse_args()
    if args.steps<10000 or args.steps%1000 or args.chunk!=1000:
        parser.error('use >=10000 steps, a multiple of1000, and chunk1000 for this protocol')
    raw=(Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_bytes();text=raw.decode()
    chars=sorted(set(text));enc={c:i for i,c in enumerate(chars)}
    data=np.array([enc[c] for c in text],np.int32)
    train,val,test=np.split(data,[int(len(data)*.8),int(len(data)*.9)])
    checkpoints=sorted({s for s in [10000,30000,args.steps] if s<=args.steps})
    transformer=args.architecture=='transformer'
    sigmoid=args.architecture=='sigmoid'
    token_product=args.architecture=='token_product'
    shallow=args.architecture in ['shallow','sigmoid','token_product']
    prefix='token-product100k' if token_product else 'longer-transformer100k' if transformer else 'longer-sigmoid100k' if sigmoid else 'longer-shallow100k' if shallow else 'longer100k'
    baseline='depth100k-transformer-seed-{seed}.json' if transformer else None if sigmoid or token_product else 'mixture100k-n1-seed-{seed}.json' if shallow else 'large-ours-seed-{seed}.json'
    config=dict(architecture=args.architecture,seed=args.seed,steps=args.steps,checkpoints=checkpoints,batch=64,lr=.001,
        optimizer='Adam beta1=.9 beta2=.999 epsilon1e-8; no schedule/weight decay',
        width=64,hidden=200 if transformer else 576 if shallow else 120,hidden_layers=1 if shallow or transformer else 6,blocks=2 if transformer else 0,residual=transformer,final_norm=transformer,
        positions='learned',pooled_norm='not applicable' if transformer else 'none' if sigmoid else 'non-learned epsilon1e-5',activation='sigmoid' if sigmoid else 'GELU',
        parameters=dict(query=99056 if transformer else 100000 if shallow else 99184,language=96081 if transformer else 81153 if shallow else 94473),tasks=args.tasks,
        training='compiled scans of1000 updates; batches generated/loaded per scan; continuous model/Adam state',
        evaluation='same fixed samples as original100k runs; predeclared update counts, no best-step selection',
        dataset_sha256=hashlib.sha256(raw).hexdigest(),baseline=baseline)
    if token_product:
        config['pooling']='sum(position[i] * token[i] * token[last]) for i < last; normalize then GELU MLP'
    print(json.dumps(config),flush=True)
    directory=Path(f'{prefix}-seed-{args.seed}');directory.mkdir(exist_ok=True)
    output=Path(f'{prefix}-seed-{args.seed}.json');runs=[]
    for task in args.tasks:
        model=TokenProductModel(task,args.seed) if token_product else TransformerModel(task,args.seed,'transformer') if transformer else SigmoidModel(task,args.seed) if sigmoid else ShallowModel(task,args.seed,'n1') if shallow else DeepModel(task,args.seed,'ours')
        assert count(model)==config['parameters']['language' if task=='language' else 'query']
        rng=np.random.default_rng(args.seed if task=='language' else 10000*args.seed+1601 if task=='position' else 50000+args.seed)
        monitor,heldout=evaluation_data(task,val,test)
        state,run=compiled_training(model,optax.adam(.001))
        history=[];checkpoints_out=[]
        result=dict(task=task,seed=args.seed,parameters=count(model),history=history,checkpoints=checkpoints_out,complete=False)
        runs.append(result)
        for start in range(0,args.steps,args.chunk):
            pairs=[training_batch(task,rng,train) for _ in range(args.chunk)]
            xs,ys=jax.device_put(np.stack([p[0] for p in pairs])),jax.device_put(np.stack([p[1] for p in pairs]))
            state,losses=run(state,xs,ys);jax.block_until_ready((state,losses))
            end=start+args.chunk
            if end%5000==0 or end in checkpoints:
                nnx.update(model,state[0]);point=metrics(model,*monitor)|dict(step=end,last_train_loss=float(losses[-1]))
                history.append(point);print(task,args.seed,end,point,flush=True)
            if end in checkpoints:
                final=evaluate(model,*heldout);record=dict(step=end,final=final)
                if end==10000 and baseline is not None:
                    prior=json.loads(Path(baseline.format(seed=args.seed)).read_text())
                    expected=next(r['final'] for r in prior['runs'] if r['task']==task)
                    np.testing.assert_allclose(final['loss'],expected['loss'],rtol=1e-4,atol=1e-5)
                    np.testing.assert_allclose(final['accuracy'],expected['accuracy'],rtol=0,atol=1e-6)
                    record['baseline_check']='10k loss/accuracy reproduce prior run within tolerance'
                checkpoints_out.append(record);save_weights(model,directory/f'{task}-step-{end}.npz')
                print('HELDOUT',task,args.seed,end,final,flush=True)
            result['updates_completed']=end
            result['complete']=end==args.steps
            # Save progress after each chunk; weights at evaluation milestones only.
            output.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
        result['final']=checkpoints_out[-1]['final']
        output.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')

if __name__=='__main__':main()
