"""Paired compiled-loop test of parameter-free hidden residual connections."""
import argparse
import json
from pathlib import Path
import time
import optax
from benchmark import jax,jnp,np,nnx,count
from budget_sweep import BatchedCandidate
from capabilities import generate
from readout_experiments import learn,metrics,save_weights
from speed_benchmark import compiled_training,synchronize_training


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--output',default='residual-seed-0.json')
    args=p.parse_args()
    if args.steps<5:p.error('at least five updates required for parity check')
    config=vars(args)|dict(context=16,vocab=96,width=64,hidden=12,hidden_layers=6,parameters=9976,
        budget=10000,batch=64,lr=.001,evaluation_seed=700016,test_examples=4096,
        validation_seed=20261122,validation_examples=16384,
        training_seed_formula='10000*seed+100*context+1',
        residual_rule='h <- h + GELU(W*h+b) at five equal-width hidden transitions; no scaling or extra normalization',
        training_mode='compiled lax.scan for both control and residual; identical pre-generated batches')
    print(json.dumps(config),flush=True)
    output=Path(args.output)
    artifacts=output.with_suffix('')
    artifacts.mkdir(exist_ok=True)
    rng=np.random.default_rng(10000*args.seed+1601)
    host_x=np.empty((args.steps,64,16),np.int32)
    host_y=np.empty((args.steps,64),np.int32)
    for i in range(args.steps):host_x[i],host_y[i],_=generate('indexed_copy',16,64,rng)
    xs,ys=jax.device_put(host_x),jax.device_put(host_y)
    jax.block_until_ready((xs,ys))
    test_x,test_y,_=generate('indexed_copy',16,4096,np.random.default_rng(700016))
    val_x,val_y,_=generate('indexed_copy',16,16384,np.random.default_rng(20261122))
    plain=BatchedCandidate(16,96,64,12,args.seed,6,False)
    residual=BatchedCandidate(16,96,64,12,args.seed,6,True)
    for a,b in zip(jax.tree.leaves(nnx.state(plain,nnx.Param)),jax.tree.leaves(nnx.state(residual,nnx.Param))):
        assert np.array_equal(np.asarray(a),np.asarray(b)),'Initial weights differ'
    tx=optax.adam(.001)
    runs=[]
    variants=[('plain',plain),('residual',residual)] if args.seed%2==0 else [('residual',residual),('plain',plain)]
    for name,model in variants:
        assert count(model)==9976
        state,run=compiled_training(model,tx)
        small_state,small_losses=run(state,xs[:5],ys[:5])
        jax.block_until_ready((small_state,small_losses))
        manual=nnx.clone(model)
        opt=nnx.Optimizer(manual,tx,wrt=nnx.Param)
        losses=[]
        for i in range(5):losses.append(learn(manual,opt,xs[i],ys[i]))
        synchronize_training(manual,opt,losses[-1])
        np.testing.assert_allclose(np.asarray(small_losses),np.asarray(losses),rtol=1e-4,atol=1e-5)
        for a,b in zip(jax.tree.leaves(small_state),jax.tree.leaves(nnx.state((manual,opt)))):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-4,atol=1e-5)
        history=[]
        begin=0
        for end in sorted(set([min(1000,args.steps),min(3000,args.steps),args.steps])):
            x,y=xs[begin:end],ys[begin:end]
            start=time.perf_counter()
            compiled=run.lower(state,x,y).compile()
            compile_seconds=time.perf_counter()-start
            start=time.perf_counter()
            state,losses=compiled(state,x,y)
            jax.block_until_ready((state,losses))
            seconds=time.perf_counter()-start
            nnx.update(model,state[0])
            point=metrics(model,test_x,test_y)|dict(step=end,compile_seconds=compile_seconds,
                execution_seconds=seconds,last_train_loss=float(losses[-1]))
            history.append(point)
            print(name,'seed',args.seed,'step',end,'accuracy',point['accuracy'],flush=True)
            begin=end
        save_weights(model,artifacts/f'{name}-seed-{args.seed}.npz')
        result=dict(variant=name,seed=args.seed,parameters=count(model),history=history,
                    final=history[-1],validation=metrics(model,val_x,val_y),
                    parity_check='five losses and complete model/Adam states match ordinary updates')
        runs.append(result)
        output.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')


if __name__=='__main__':main()
