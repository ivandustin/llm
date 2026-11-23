"""CPU timing of current token-product and sinusoidal Transformer checkpoints.

Run from repository root with python -m experiments.current_speed --task TASK.
Shared timing helpers and five-update Adam parity validation follow the historical
benchmark. No checkpoint files are modified. See JSON for environment and trials.
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import time
import optax
from experiments.benchmark import jax,jnp,np,nnx,count
from experiments.longer_training_100k import TokenProductModel,training_batch
from experiments.depth_100k_test import Model as TransformerModel
from experiments.validate_layer_depth import load_weights
from experiments.readout_experiments import learn
from experiments.speed_benchmark import inference,synchronize_training,compiled_training,summarize


def make_bank(batch,task,n=32):
    rng=np.random.default_rng(421000+batch)
    text=Path('data/shakespeare.txt').read_text()
    enc={c:i for i,c in enumerate(sorted(set(text)))}
    train=np.array([enc[c] for c in text[:int(len(text)*.8)]],np.int32)
    bank=[]
    for _ in range(n):
        x,y=training_batch(task,rng,train)
        bank.append((jax.device_put(x[:batch]),jax.device_put(y[:batch])))
    jax.block_until_ready(bank)
    return bank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=7)
    p.add_argument('--train-steps',type=int,default=200)
    p.add_argument('--infer-steps',type=int,default=500)
    p.add_argument('--scan-steps',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',default='current-speed-results.json')
    p.add_argument('--task',choices=['position','single','multi','language'],default='position')
    p.add_argument('--checkpoint-root',type=Path,default=Path('experiments/current64-50k'))
    args=p.parse_args()
    if min(args.trials,args.train_steps,args.infer_steps,args.scan_steps,args.warmup)<1:p.error('counts must be positive')
    cpuinfo=Path('/proc/cpuinfo').read_text()
    cpu=next(line.split(':',1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('model name'))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}|dict(cpu=cpu,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),
        devices=[str(d) for d in jax.devices()],software={n:importlib.metadata.version(n) for n in ['jax','jaxlib','flax','optax','numpy']},
        environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS']},
        transformer_blocks=2,transformer_width=64,transformer_ff_hidden=200,candidate_width=64,candidate_hidden=576,candidate_hidden_layers=1,positions='sinusoidal',context=32 if args.task=='language' else 16,vocab=65 if args.task=='language' else 96,training_batch=64,inference_batches=[1,64],checkpoint_seed=0,checkpoint_step=50000,lr=.001,
        timing_note='One process; no competing experiment jobs; alternating model order; resident FP32 model and int32 inputs; data generation and host transfers excluded. Training optimizer states start fresh; timing updates discarded.')
    candidate=TokenProductModel(args.task,0)
    transformer=TransformerModel(args.task,0,'transformer',positions='sinusoidal')
    models=dict(candidate=candidate,transformer=transformer)
    for name,model in models.items():
        architecture='token_product' if name=='candidate' else 'transformer'
        load_weights(model,args.checkpoint_root/f'{architecture}-seed-0'/f'{args.task}-step-50000.npz')
    config['parameters']={n:count(m) for n,m in models.items()}
    expected=dict(candidate=81153,transformer=94033) if args.task=='language' else dict(candidate=100000,transformer=98032)
    assert config['parameters']==expected
    results={}
    compilation={}
    banks={b:make_bank(b,args.task) for b in [1,64]}
    def order(t):
        names=list(models)
        return names if t%2==0 else names[::-1]
    print(json.dumps(config),flush=True)
    # Request latency, including JIT dispatch and output synchronization.
    for batch in [1,64]:
        bank=banks[batch]
        mode=f'inference_batch_{batch}'
        samples={n:[] for n in models}
        for name,model in models.items():
            t=time.perf_counter()
            jax.block_until_ready(inference(model,bank[0][0]))
            compilation[f'{mode}_{name}_first_call_seconds']=time.perf_counter()-t
            for i in range(args.warmup):jax.block_until_ready(inference(model,bank[i%len(bank)][0]))
        for trial in range(args.trials):
            for name in order(trial):
                t=time.perf_counter()
                for i in range(args.infer_steps):
                    jax.block_until_ready(inference(models[name],bank[i%len(bank)][0]))
                samples[name].append((time.perf_counter()-t)/args.infer_steps)
        results[mode]={n:summarize(samples[n],batch) for n in models}
        print(mode,json.dumps(results[mode]),flush=True)
    # Python-loop training includes dispatch, not random sample generation.
    bank=banks[64]
    tx=optax.adam(.001)
    samples={n:[] for n in models}
    for name,base in models.items():
        m=nnx.clone(base)
        o=nnx.Optimizer(m,tx,wrt=nnx.Param)
        t=time.perf_counter()
        loss=learn(m,o,*bank[0])
        synchronize_training(m,o,loss)
        compilation[f'python_training_{name}_first_call_seconds']=time.perf_counter()-t
    for trial in range(args.trials):
        for name in order(trial):
            m=nnx.clone(models[name])
            o=nnx.Optimizer(m,tx,wrt=nnx.Param)
            for i in range(args.warmup):loss=learn(m,o,*bank[i%len(bank)])
            synchronize_training(m,o,loss)
            t=time.perf_counter()
            for i in range(args.train_steps):loss=learn(m,o,*bank[i%len(bank)])
            synchronize_training(m,o,loss)
            samples[name].append((time.perf_counter()-t)/args.train_steps)
    results['python_training_batch_64']={n:summarize(samples[n],64) for n in models}
    print('python_training_batch_64',json.dumps(results['python_training_batch_64']),flush=True)
    # Compiled loop removes Python dispatch between updates. Inputs remain
    # dynamic arguments; model/Adam state is carried through every update.
    xs=jnp.stack([bank[i%len(bank)][0] for i in range(args.scan_steps)])
    ys=jnp.stack([bank[i%len(bank)][1] for i in range(args.scan_steps)])
    jax.block_until_ready((xs,ys))
    executables={}
    for name,base in models.items():
        state,run=compiled_training(nnx.clone(base),tx)
        # Check five updates against the existing training implementation.
        small_x=jnp.stack([bank[i%len(bank)][0] for i in range(5)])
        small_y=jnp.stack([bank[i%len(bank)][1] for i in range(5)])
        scan_state,scan_losses=run(state,small_x,small_y)
        jax.block_until_ready((scan_state,scan_losses))
        m=nnx.clone(base)
        o=nnx.Optimizer(m,tx,wrt=nnx.Param)
        manual_losses=[]
        for i in range(5):manual_losses.append(learn(m,o,*bank[i%len(bank)]))
        synchronize_training(m,o,manual_losses[-1])
        np.testing.assert_allclose(np.asarray(scan_losses),np.asarray(manual_losses),rtol=1e-4,atol=1e-5)
        for a,b in zip(jax.tree.leaves(scan_state),jax.tree.leaves(nnx.state((m,o)))):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-4,atol=1e-5)
        t=time.perf_counter()
        compiled=run.lower(state,xs,ys).compile()
        compilation[f'compiled_training_{name}_trace_compile_seconds']=time.perf_counter()-t
        jax.block_until_ready(compiled(state,xs,ys))
        executables[name]=(state,compiled)
    samples={n:[] for n in models}
    for trial in range(args.trials):
        for name in order(trial):
            state,compiled=executables[name]
            t=time.perf_counter()
            output=compiled(state,xs,ys)
            jax.block_until_ready(output)
            samples[name].append((time.perf_counter()-t)/args.scan_steps)
    results['compiled_training_batch_64']={n:summarize(samples[n],64) for n in models}
    print('compiled_training_batch_64',json.dumps(results['compiled_training_batch_64']),flush=True)
    ratios={mode:values['transformer']['milliseconds_per_step_median']/values['candidate']['milliseconds_per_step_median'] for mode,values in results.items()}
    output=dict(config=config,results=results,candidate_speedup=ratios,compilation=compilation,
                validation='Compiled training losses and all model/Adam state leaves match five ordinary updates within rtol 1e-4 / atol 1e-5.')
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print('candidate speedup',json.dumps(ratios),flush=True)


if __name__=='__main__':main()
