"""Wider normalized-GELU models under a strict total-parameter budget."""
import argparse
import json
from pathlib import Path

from experiments.candidate_model import Model as Candidate
from experiments.benchmark import nnx, np, jnp, count
from experiments.capabilities import ProbeTransformer, generate
from experiments.readout_experiments import learn, metrics, save_weights
import optax


class BatchedCandidate(nnx.Module):
    def __init__(self,context,vocab,width,hidden,seed,hidden_layers=1,residual=False,residual_prenorm=False):
        ModelType=Candidate
        if hidden_layers==2:
            from experiments.deep_candidate_model import Model as ModelType
        if hidden_layers>2 or residual:
            from experiments.stacked_candidate_model import Model as Stacked
            self.model=Stacked([context,vocab],vocab,width,hidden,nnx.Rngs(seed),hidden_layers,residual=residual,residual_prenorm=residual_prenorm)
        else:
            self.model=ModelType([context,vocab],vocab,width,hidden,nnx.Rngs(seed))

    def __call__(self,tokens):
        positions=jnp.broadcast_to(jnp.arange(tokens.shape[-1]),tokens.shape)
        return nnx.vmap(self.model)(jnp.stack([positions,tokens],axis=1))


def hidden_for_budget(context,vocab,width,budget,hidden_layers=1):
    # Two embedding tables plus Linear(width,hidden) and Linear(hidden,vocab).
    h=(budget-(context+vocab)*width-vocab)//(width+vocab+1)
    if hidden_layers>1:
        while h>0 and (context+vocab)*width+(width+vocab+1)*h+vocab+(hidden_layers-1)*(h*h+h)>budget:
            h-=1
    return h


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--widths',type=int,nargs='+',default=[16,24,32,40,48,56,64,72,80])
    p.add_argument('--budget',type=int,default=10000)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--output',default='budget-results.json')
    p.add_argument('--hidden-layers',type=int,choices=range(1,9),default=1)
    p.add_argument('--skip-reference',action='store_true')
    p.add_argument('--residual',action='store_true')
    args=p.parse_args()
    if args.residual and args.hidden_layers<2:p.error('residual connections require at least two hidden layers')
    context,vocab=16,96
    if args.steps<1 or args.seeds<1:p.error('steps and seeds must be positive')
    if any(w<1 or hidden_for_budget(context,vocab,w,args.budget,args.hidden_layers)<1 for w in args.widths):
        p.error('each width must leave budget for at least one hidden unit')
    config=vars(args)|dict(context=context,vocab=vocab,batch=64,lr=.001,test_examples=4096,
        task='indexed_copy',variant='normalized_GELU',transformer_depth=1,
        evaluation_seed=700000+context,training_seed_formula='10000*seed+100*context+1')
    path=Path(args.output)
    artifacts=path.with_suffix('')
    artifacts.mkdir(exist_ok=True)
    test_x,test_y,_=generate('indexed_copy',context,4096,np.random.default_rng(config['evaluation_seed']))
    reference_base=count(ProbeTransformer(vocab,context,16,nnx.Rngs(0),depth=1,ff_hidden=1))
    reference_hidden=1+(args.budget-reference_base)//33
    if reference_hidden<1:p.error('budget too small for transformer reference')
    specs=[('candidate',w,hidden_for_budget(context,vocab,w,args.budget,args.hidden_layers)) for w in args.widths]
    if not args.skip_reference: specs.append(('transformer',16,reference_hidden))
    print(json.dumps(config),flush=True)
    runs=[]
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for name,width,hidden in specs:
            model=(BatchedCandidate(context,vocab,width,hidden,seed,args.hidden_layers,args.residual) if name=='candidate' else
                   ProbeTransformer(vocab,context,width,nnx.Rngs(seed),depth=1,ff_hidden=hidden))
            parameters=count(model)
            assert parameters<=args.budget
            if name=='candidate':
                assert parameters==(context+vocab)*width+(width+vocab+1)*hidden+vocab+(args.hidden_layers-1)*(hidden*hidden+hidden)
            optimizer=nnx.Optimizer(model,optax.adam(.001),wrt=nnx.Param)
            rng=np.random.default_rng(10000*seed+100*context+1)
            history=[]
            for i in range(args.steps):
                x,y,_=generate('indexed_copy',context,64,rng)
                loss=float(learn(model,optimizer,x,y))
                if i+1 in sorted(set([min(1000,args.steps),min(3000,args.steps),args.steps])):
                    point=metrics(model,test_x,test_y)|dict(step=i+1,train_loss=loss)
                    history.append(point)
                    print(f'{name} width={width} hidden={hidden} params={parameters} seed={seed} step={i+1} accuracy={point["accuracy"]:.4f}',flush=True)
            save_weights(model,artifacts/f'{name}-width-{width}-seed-{seed}.npz')
            runs.append(dict(model=name,width=width,hidden=hidden,parameters=parameters,seed=seed,
                             history=history,final=history[-1]))
            path.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')


if __name__=='__main__':main()
