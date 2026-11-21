"""Next-character comparison of the retained six-layer architecture.

Uses the original language benchmark's dataset, splits, training batches and
fixed evaluation windows. All models train in compiled loops with matched
parameter counts. The six-layer architecture is kept at width 64 / hidden 12.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import optax
from experiments.benchmark import nnx,np,jnp,jax,count,batch
from experiments.best_model import Batched as Best
from experiments.budget_sweep import BatchedCandidate
from experiments.capabilities import ProbeTransformer
from experiments.readout_experiments import learn,metrics,save_weights
from experiments.speed_benchmark import compiled_training,synchronize_training,inference


def evaluate(model,x,y):
    result=metrics(model,x,y)
    result['perplexity']=float(np.exp(result['loss']))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--output',default='language-depth-seed-0.json')
    p.add_argument('--deep-width',type=int,default=64)
    p.add_argument('--deep-hidden',type=int,default=12)
    p.add_argument('--deep-layers',type=int,default=6)
    p.add_argument('--residual',action='store_true')
    p.add_argument('--residual-prenorm',action='store_true')
    p.add_argument('--models',nargs='+',choices=['deep','shallow','transformer'],default=['deep','shallow','transformer'])
    args=p.parse_args()
    if args.steps<5:p.error('at least five updates required for parity check')
    if args.deep_width<1 or args.deep_hidden<1:p.error('width and hidden units must be positive')
    if args.residual_prenorm and not args.residual:p.error("prenorm requires --residual")
    context=32
    raw=(Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_bytes()
    text=raw.decode()
    chars=sorted(set(text))
    encoding={c:i for i,c in enumerate(chars)}
    data=np.array([encoding[c] for c in text],np.int32)
    train,validation,test=np.split(data,[int(len(data)*.8),int(len(data)*.9)])
    vocab=len(chars)
    assert vocab==65
    config=vars(args)|dict(context=context,vocab=vocab,batch=64,lr=.001,
        dataset_sha256=hashlib.sha256(raw).hexdigest(),split_lengths=list(map(len,[train,validation,test])),
        evaluation_seed=8675309,evaluation_examples_per_split=4096,
        deep=dict(width=args.deep_width,hidden=args.deep_hidden,hidden_layers=args.deep_layers,residual=args.residual,residual_prenorm=args.residual_prenorm,
            parameters=(context+vocab)*args.deep_width+(args.deep_width+vocab+1)*args.deep_hidden+vocab+(args.deep_layers-1)*(args.deep_hidden**2+args.deep_hidden)),
        shallow=dict(width=64,hidden=18,hidden_layers=1,parameters=8613),
        transformer=dict(width=16,heads=4,blocks=1,ff_hidden=144,parameters=8609),
        training_mode='compiled lax.scan for all models; initial weights random and independently seeded',
        checkpoint_rule='report final update; no validation-based checkpoint selection')
    print(json.dumps(config),flush=True)
    output=Path(args.output)
    artifact_dir=output.with_suffix('')
    artifact_dir.mkdir(exist_ok=True)
    fixed=np.random.default_rng(8675309)
    eval_data={}
    for name,part in [('validation',validation),('test',test)]:
        batches=[batch(part,fixed.integers(0,len(part)-context,size=128),context) for _ in range(32)]
        eval_data[name]=(np.concatenate([b[0] for b in batches]),np.concatenate([b[1] for b in batches]))
    rng=np.random.default_rng(args.seed)
    host_x=np.empty((args.steps,64,context),np.int32)
    host_y=np.empty((args.steps,64),np.int32)
    for i in range(args.steps):
        host_x[i],host_y[i]=batch(train,rng.integers(0,len(train)-context,size=64),context)
    xs,ys=jax.device_put(host_x),jax.device_put(host_y)
    jax.block_until_ready((xs,ys))
    def make_deep():
        if args.deep_width==64 and args.deep_hidden==12 and args.deep_layers==6 and not args.residual:return Best(context,vocab,args.seed)
        return BatchedCandidate(context,vocab,args.deep_width,args.deep_hidden,args.seed,args.deep_layers,args.residual,args.residual_prenorm)
    specs=[('deep',make_deep),
           ('shallow',lambda:BatchedCandidate(context,vocab,64,18,args.seed,1)),
           ('transformer',lambda:ProbeTransformer(vocab,context,16,nnx.Rngs(args.seed),depth=1,ff_hidden=144))]
    tx=optax.adam(.001)
    results=[]
    for name,create in specs:
        if name not in args.models:continue
        model=create()
        assert count(model)==config[name]['parameters'] and count(model)<=10000
        state,run=compiled_training(model,tx)
        small_state,small_losses=run(state,xs[:5],ys[:5])
        jax.block_until_ready((small_state,small_losses))
        manual=nnx.clone(model)
        optimizer=nnx.Optimizer(manual,tx,wrt=nnx.Param)
        manual_losses=[]
        for i in range(5):manual_losses.append(learn(manual,optimizer,xs[i],ys[i]))
        synchronize_training(manual,optimizer,manual_losses[-1])
        np.testing.assert_allclose(np.asarray(small_losses),np.asarray(manual_losses),rtol=1e-4,atol=1e-5)
        for a,b in zip(jax.tree.leaves(small_state),jax.tree.leaves(nnx.state((manual,optimizer)))):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-4,atol=1e-5)
        begin=0
        history=[]
        for end in sorted(set([min(1000,args.steps),min(3000,args.steps),args.steps])):
            x,y=xs[begin:end],ys[begin:end]
            start=time.perf_counter()
            executable=run.lower(state,x,y).compile()
            compile_seconds=time.perf_counter()-start
            start=time.perf_counter()
            state,losses=executable(state,x,y)
            jax.block_until_ready((state,losses))
            execution_seconds=time.perf_counter()-start
            nnx.update(model,state[0])
            point=dict(step=end,validation=evaluate(model,*eval_data['validation']),
                final_train_loss=float(np.asarray(losses)[-100:].mean()),compile_seconds=compile_seconds,
                execution_seconds=execution_seconds)
            history.append(point)
            print(name,'seed',args.seed,'step',end,'val_perplexity',point['validation']['perplexity'],flush=True)
            begin=end
        save_weights(model,artifact_dir/f'{name}-seed-{args.seed}.npz')
        result=dict(model=name,seed=args.seed,parameters=count(model),history=history,
                    validation=history[-1]['validation'],test=evaluate(model,*eval_data['test']),
                    implementation_check='five-step losses and all model/Adam states match ordinary updates')
        prompt='First Citizen:\n'
        sequence=[encoding[c] for c in ('\n'*context+prompt)]
        key=jax.random.key(args.seed)
        for _ in range(160):
            key,subkey=jax.random.split(key)
            scores=inference(model,jnp.array([sequence[-context:]],dtype=jnp.int32))[0]
            sequence.append(int(jax.random.categorical(subkey,scores/.8)))
        result['sample']=''.join(chars[i] for i in sequence[context:])
        results.append(result)
        print(name,'seed',args.seed,'test',json.dumps(result['test']),flush=True)
        output.write_text(json.dumps(dict(config=config,runs=results),indent=2)+'\n')


if __name__=='__main__':main()
