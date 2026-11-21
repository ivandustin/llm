"""Longer chunked training of deep or shallow pooling, with10k parity checks."""
import argparse
import hashlib
import json
from pathlib import Path

import optax
from benchmark import nnx,np,jax,count,batch
from benchmark_100k import Model as DeepModel
from parallel_mixture_100k import Model as ShallowModel
from capabilities import generate
from mqar_test import generate as associative
from language_depth_test import evaluate
from readout_experiments import metrics,save_weights
from speed_benchmark import compiled_training

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
    parser.add_argument('--architecture',choices=['deep','shallow'],default='deep')
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--chunk',type=int,default=1000)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    args=parser.parse_args()
    if args.steps<10000 or args.steps%1000 or args.chunk!=1000:
        parser.error('use >=10000 steps, a multiple of1000, and chunk1000 for this protocol')
    raw=Path('data/shakespeare.txt').read_bytes();text=raw.decode()
    chars=sorted(set(text));enc={c:i for i,c in enumerate(chars)}
    data=np.array([enc[c] for c in text],np.int32)
    train,val,test=np.split(data,[int(len(data)*.8),int(len(data)*.9)])
    checkpoints=sorted({s for s in [10000,30000,args.steps] if s<=args.steps})
    shallow=args.architecture=='shallow'
    prefix='longer-shallow100k' if shallow else 'longer100k'
    baseline='mixture100k-n1-seed-{seed}.json' if shallow else 'large-ours-seed-{seed}.json'
    config=dict(architecture=args.architecture,seed=args.seed,steps=args.steps,checkpoints=checkpoints,batch=64,lr=.001,
        optimizer='Adam beta1=.9 beta2=.999 epsilon1e-8; no schedule/weight decay',
        width=64,hidden=576 if shallow else 120,hidden_layers=1 if shallow else 6,residual=False,final_norm=False,
        positions='learned',pooled_norm='non-learned epsilon1e-5',activation='GELU',
        parameters=dict(query=100000 if shallow else 99184,language=81153 if shallow else 94473),tasks=args.tasks,
        training='compiled scans of1000 updates; batches generated/loaded per scan; continuous model/Adam state',
        evaluation='same fixed samples as original100k runs; predeclared update counts, no best-step selection',
        dataset_sha256=hashlib.sha256(raw).hexdigest(),baseline=baseline)
    print(json.dumps(config),flush=True)
    directory=Path(f'{prefix}-seed-{args.seed}');directory.mkdir(exist_ok=True)
    output=Path(f'{prefix}-seed-{args.seed}.json');runs=[]
    for task in args.tasks:
        model=ShallowModel(task,args.seed,'n1') if shallow else DeepModel(task,args.seed,'ours')
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
                if end==10000:
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
