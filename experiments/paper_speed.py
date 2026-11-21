"""Isolated, synchronized CPU timing of the selected trained models.

One process, shared CPU affinity and inputs, alternating model order. Measures
per-request inference and both Python-driven and compiled-loop Adam updates.
Checkpoint accuracy is not changed or re-evaluated by these timing updates.
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
from experiments.budget_sweep import BatchedCandidate
from experiments.benchmark_100k import Model as LargeModel
from experiments.capabilities import ProbeTransformer,generate
from experiments.readout_experiments import learn
from experiments.validate_layer_depth import load_weights


@nnx.jit
def inference(model,x):
    return model(x)


def synchronize_training(model,optimizer,loss):
    # Wait for all updated arrays, not just loss (which precedes the update).
    jax.block_until_ready((loss,nnx.state(model),nnx.state(optimizer)))


def make_bank(batch,n=32):
    rng=np.random.default_rng(421000+batch)
    bank=[]
    for _ in range(n):
        x,y,_=generate('indexed_copy',16,batch,rng)
        bank.append((jax.device_put(x),jax.device_put(y)))
    jax.block_until_ready(bank)
    return bank


def compiled_training(model,tx):
    optimizer=nnx.Optimizer(model,tx,wrt=nnx.Param)
    graph,state=nnx.split((model,optimizer))
    def body(carry,xy):
        m,o=nnx.merge(graph,carry)
        x,y=xy
        def loss_fn(mod):
            return optax.softmax_cross_entropy_with_integer_labels(mod(x),y).mean()
        loss,grads=nnx.value_and_grad(loss_fn)(m)
        o.update(m,grads)
        return nnx.state((m,o)),loss
    @jax.jit
    def run(state,xs,ys):
        return jax.lax.scan(body,state,(xs,ys))
    return state,run


def summarize(samples,batch):
    values=np.array(samples)
    med=float(np.median(values))
    return dict(milliseconds_per_step_median=1000*med,
        milliseconds_per_step_min=1000*float(values.min()),
        milliseconds_per_step_max=1000*float(values.max()),
        milliseconds_per_step_sd=1000*float(values.std(ddof=1)) if len(values)>1 else 0.,
        examples_per_second=batch/med,seconds_per_step_trials=samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=7)
    p.add_argument('--train-steps',type=int,default=200)
    p.add_argument('--infer-steps',type=int,default=500)
    p.add_argument('--scan-steps',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',default='paper-speed-results.json')
    p.add_argument('--include-residual',action='store_true')
    args=p.parse_args()
    if min(args.trials,args.train_steps,args.infer_steps,args.scan_steps,args.warmup)<1:p.error('counts must be positive')
    cpuinfo=Path('/proc/cpuinfo').read_text()
    cpu=next(line.split(':',1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('model name'))
    config=vars(args)|dict(cpu=cpu,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),
        devices=[str(d) for d in jax.devices()],software={n:importlib.metadata.version(n) for n in ['jax','jaxlib','flax','optax','numpy']},
        environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS']},
        context=16,vocab=96,training_batch=64,inference_batches=[1,64],checkpoint_seed=0,lr=.001,
        timing_note='One process; no competing experiment jobs; alternating order for two models or rotating/reversing order for three; resident FP32 model and int32 inputs; data generation and host transfers excluded. Training optimizer states start fresh; timing updates discarded.')
    candidate=LargeModel('position',0,'ours')
    load_weights(candidate,Path('large-ours-seed-0')/'position.npz')
    transformer=LargeModel('position',0,'transformer')
    load_weights(transformer,Path('large-transformer-seed-0')/'position.npz')
    if args.include_residual:
        residual=BatchedCandidate(16,96,64,12,0,6,True)
        load_weights(residual,Path('residual-seed-0')/'residual-seed-0.npz')
        models=dict(candidate=candidate,residual=residual,transformer=transformer)
    else:
        models=dict(candidate=candidate,transformer=transformer)
    config['parameters']={n:count(m) for n,m in models.items()}
    expected=dict(candidate=99184,transformer=99640)
    if args.include_residual:expected['residual']=9976
    assert config['parameters']==expected
    results={}
    compilation={}
    banks={b:make_bank(b) for b in [1,64]}
    def order(t):
        names=list(models)
        if len(names)==2:return names if t%2==0 else names[::-1]
        shift=t%len(names)
        names=names[shift:]+names[:shift]
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
    if args.include_residual:
        output['residual_speedup']={mode:values['transformer']['milliseconds_per_step_median']/values['residual']['milliseconds_per_step_median'] for mode,values in results.items()}
        output['residual_vs_candidate']={mode:values['candidate']['milliseconds_per_step_median']/values['residual']['milliseconds_per_step_median'] for mode,values in results.items()}
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print('candidate speedup',json.dumps(ratios),flush=True)


if __name__=='__main__':main()
