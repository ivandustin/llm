"""Test only sigmoid query gating, one hidden layer, at <=10k parameters."""
import argparse
import json
from pathlib import Path
from gated_candidate_model import Model
from benchmark import nnx,np,jnp,count
from capabilities import generate
from readout_experiments import learn,metrics,save_weights
import optax


class Batched(nnx.Module):
    def __init__(self,width,hidden,seed):
        self.model=Model([16,96],96,width,hidden,nnx.Rngs(seed))

    def __call__(self,tokens):
        positions=jnp.broadcast_to(jnp.arange(tokens.shape[-1]),tokens.shape)
        return nnx.vmap(self.model)(jnp.stack([positions,tokens],axis=1))


def hidden_for_budget(width,budget):
    return (budget-112*width-96)//(2*width+98)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--widths',type=int,nargs='+',default=[16,24,32,40,48,56,64,72,80])
    p.add_argument('--budget',type=int,default=10000)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--output',default='gated-budget-results.json')
    args=p.parse_args()
    if args.steps<1 or args.seeds<1:p.error('steps and seeds must be positive')
    if any(w<1 or hidden_for_budget(w,args.budget)<1 for w in args.widths):p.error('insufficient budget for these widths')
    config=vars(args)|dict(context=16,vocab=96,batch=64,lr=.001,test_examples=4096,task='indexed_copy',
        variant='normalized_GELU_sigmoid_query_gate',hidden_layers=1,query='shared embedding of last input token',
        query_normalization=False,evaluation_seed=700016,training_seed_formula='10000*seed+100*context+1')
    output=Path(args.output)
    artifacts=output.with_suffix('')
    artifacts.mkdir(exist_ok=True)
    x_test,y_test,_=generate('indexed_copy',16,4096,np.random.default_rng(config['evaluation_seed']))
    runs=[]
    print(json.dumps(config),flush=True)
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for width in args.widths:
            hidden=hidden_for_budget(width,args.budget)
            model=Batched(width,hidden,seed)
            params=count(model)
            assert params==112*width+(2*width+98)*hidden+96 and params<=args.budget
            opt=nnx.Optimizer(model,optax.adam(.001),wrt=nnx.Param)
            rng=np.random.default_rng(10000*seed+1601)
            history=[]
            for i in range(args.steps):
                x,y,_=generate('indexed_copy',16,64,rng)
                loss=float(learn(model,opt,x,y))
                if i+1 in sorted(set([min(1000,args.steps),min(3000,args.steps),args.steps])):
                    point=metrics(model,x_test,y_test)|dict(step=i+1,train_loss=loss)
                    history.append(point)
                    print(f'sigmoid_gate width={width} hidden={hidden} params={params} seed={seed} step={i+1} accuracy={point["accuracy"]:.4f}',flush=True)
            save_weights(model,artifacts/f'width-{width}-seed-{seed}.npz')
            runs.append(dict(width=width,hidden=hidden,parameters=params,seed=seed,history=history,final=history[-1]))
            output.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')


if __name__=='__main__':main()
