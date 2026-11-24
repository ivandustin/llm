"""Controlled capability probes; fresh generated examples for every update."""
import argparse
import json
import math
from pathlib import Path
import time

from experiments.benchmark import Model, count, step, evaluate, nnx, jnp, np
import optax


class Block(nnx.Module):
    def __init__(self, width, rngs, ff_hidden=None, ff_depth=1):
        ff_hidden=ff_hidden or 4*width
        self.n1 = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3*width, rngs=rngs)
        self.proj = nnx.Linear(width, width, rngs=rngs)
        self.n2 = nnx.LayerNorm(width, rngs=rngs)
        self.up = nnx.Linear(width, ff_hidden, rngs=rngs)
        self.down = nnx.Linear(ff_hidden, width, rngs=rngs)
        self.ff_layers=nnx.List([nnx.Linear(ff_hidden,ff_hidden,rngs=rngs) for _ in range(ff_depth-1)])

    def __call__(self, x):
        b, length, width = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, length, 3, 4, width//4)
        q, k, v = [qkv[:,:,i].transpose(0,2,1,3) for i in range(3)]
        scores = q @ k.swapaxes(-1,-2) / math.sqrt(width//4)
        mask = jnp.arange(length)[:,None] >= jnp.arange(length)[None,:]
        from jax.nn import softmax, gelu
        weights = softmax(jnp.where(mask, scores, -1e9), axis=-1)
        z = (weights @ v).transpose(0,2,1,3).reshape(b,length,width)
        x = x + self.proj(z)
        h=gelu(self.up(self.n2(x)))
        for layer in self.ff_layers:h=gelu(layer(h))
        return x+self.down(h)


class ProbeTransformer(nnx.Module):
    def __init__(self, vocab, context, width, rngs, depth=2, ff_hidden=None, ff_depth=1):
        self.token = nnx.Embed(vocab,width,rngs=rngs)
        self.position = nnx.Embed(context,width,rngs=rngs)
        self.blocks = nnx.List([Block(width,rngs,ff_hidden,ff_depth) for _ in range(depth)])
        self.norm = nnx.LayerNorm(width,rngs=rngs)
        self.head = nnx.Linear(width,vocab,rngs=rngs)

    def __call__(self, tokens):
        x = self.token(tokens) + self.position(jnp.arange(tokens.shape[-1]))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:,-1]))


TASKS = ['first_copy','indexed_copy','associative_recall','relative_order','majority']


def generate(task, context, size, rng):
    # Shared vocabulary size 96; unused output classes remain in the softmax.
    x = rng.integers(0,16,(size,context),dtype=np.int32)
    if task == 'first_copy':
        y = x[:,0].copy()
        chance = 1/16
    elif task == 'indexed_copy':
        index = rng.integers(0,context-1,size)
        y = x[np.arange(size),index].copy()
        # Distinct query tokens encode the requested zero-based position.
        x[:,-1] = 16 + index
        chance = 1/16
    elif task == 'associative_recall':
        pairs = (context-1)//2
        keys = np.stack([rng.choice(32,pairs,replace=False) for _ in range(size)])
        values = rng.integers(32,48,(size,pairs),dtype=np.int32)
        x.fill(95)
        x[:,:2*pairs:2] = keys
        x[:,1:2*pairs:2] = values
        index = rng.integers(0,pairs,size)
        x[:,-1] = keys[np.arange(size),index]
        y = values[np.arange(size),index]
        chance = 1/16
    elif task == 'relative_order':
        # Exactly one A and one B; otherwise independent distractors.
        x = rng.integers(2,16,(size,context),dtype=np.int32)
        positions = np.stack([rng.choice(context,2,replace=False) for _ in range(size)])
        x[np.arange(size),positions[:,0]] = 0
        x[np.arange(size),positions[:,1]] = 1
        y = (positions[:,0] < positions[:,1]).astype(np.int32)
        chance = .5
    elif task == 'majority':
        x = rng.integers(0,2,(size,context),dtype=np.int32)
        # Use an odd number of bits so there is no tie or class bias.
        x[:,-1] = 95
        y = (x[:,:-1].sum(-1) > (context-1)//2).astype(np.int32)
        chance = .5
    else:
        raise ValueError(task)
    return x, y, chance


def verify_generators():
    for context in [16,64]:
        for task in TASKS:
            x,y,_ = generate(task,context,256,np.random.default_rng(13))
            assert x.shape == (256,context) and x.min() >= 0 and x.max() < 96
            if task == 'first_copy': expected=x[:,0]
            elif task == 'indexed_copy': expected=x[np.arange(len(x)),x[:,-1]-16]
            elif task == 'associative_recall':
                pairs=(context-1)//2
                keys=x[:,:2*pairs:2]
                assert np.all((keys == x[:,-1,None]).sum(-1) == 1)
                expected=x[:,1:2*pairs:2][np.arange(len(x)),(keys == x[:,-1,None]).argmax(-1)]
            elif task == 'relative_order': expected=(x.argmin(-1) < np.where(x==1,np.arange(context),context).min(-1)).astype(int)
            else: expected=(x[:,:-1].sum(-1) >= context//2).astype(int)
            assert np.array_equal(y,expected),task
    print('Generator labels verified for all tasks and contexts.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--contexts',type=int,nargs='+',default=[16,64])
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    p.add_argument('--output',default='capabilities-results.json')
    p.add_argument('--models',nargs='+',choices=['cube','transformer'],default=['cube','transformer'])
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--depth',type=int,choices=[1,2],default=2)
    args=p.parse_args()
    if args.width <= 0 or args.width % 4: p.error('--width must be positive and divisible by four')
    if any(c < 4 or c > 64 or c % 2 for c in args.contexts):
        p.error('--contexts must be even lengths between 4 and 64')
    if args.steps < 1 or args.seeds < 1: p.error('--steps and --seeds must be positive')
    verify_generators()
    width=args.width
    path=Path(args.output)
    config=vars(args)|dict(vocab=96,batch=64,transformer_depth=args.depth,test_examples=4096)
    previous=json.loads(path.read_text()) if args.resume and path.exists() else None
    if previous and any(previous['config'].get(k) != config[k] for k in
                        ['steps','width','lr','vocab','batch','transformer_depth','test_examples']):
        p.error('resume configuration does not match the saved training configuration')
    results=previous['runs'] if previous else []
    for context in args.contexts:
        target=count(ProbeTransformer(96,context,width,nnx.Rngs(0)))
        hidden=min(range(1,2048),key=lambda h: abs((context+96)*width+(width+1)*h+(h+1)*96-target))
        ff_hidden=4*width
        if args.depth==1:
            # Keep the original cube size and match a single transformer block
            # by expanding only that block's feed-forward hidden layer.
            cube_count=(context+96)*width+(width+1)*hidden+(hidden+1)*96
            base=count(ProbeTransformer(96,context,width,nnx.Rngs(0),depth=1,ff_hidden=1))
            ff_hidden=min(range(1,4096),key=lambda h: abs(base+(2*width+1)*(h-1)-cube_count))
        for task_index,task in enumerate(TASKS):
            if task not in args.tasks: continue
            test_x,test_y,chance=generate(task,context,4096,np.random.default_rng(90000+100*context+task_index))
            for seed in range(args.seed_start,args.seed_start+args.seeds):
                for name in args.models:
                    if any(r['context']==context and r['task']==task and r['seed']==seed and r['model']==name for r in results):
                        continue
                    cube=name=='cube'
                    model=(Model([context,96],96,width,hidden,nnx.Rngs(seed)) if cube else
                           ProbeTransformer(96,context,width,nnx.Rngs(seed),depth=args.depth,ff_hidden=ff_hidden))
                    optimizer=nnx.Optimizer(model,optax.adam(args.lr),wrt=nnx.Param)
                    rng=np.random.default_rng(10000*seed+100*context+task_index)
                    history=[]
                    training_seconds=0.
                    for i in range(args.steps):
                        x,y,_=generate(task,context,64,rng)
                        start=time.perf_counter()
                        loss=float(step(model,optimizer,x,y,cube))
                        if i: training_seconds+=time.perf_counter()-start
                        if (i+1) in sorted(set([min(500,args.steps),min(1500,args.steps),args.steps])):
                            metrics=np.mean([[float(v) for v in evaluate(model,test_x[j:j+128],test_y[j:j+128],cube)]
                                             for j in range(0,len(test_y),128)],axis=0)
                            point=dict(step=i+1,loss=float(metrics[0]),accuracy=float(metrics[1]))
                            history.append(point)
                            print(f'{task} L={context} seed={seed} {name} step={i+1} accuracy={metrics[1]:.4f}',flush=True)
                    result=dict(task=task,context=context,seed=seed,model=name,parameters=count(model),hidden=hidden if cube else ff_hidden,
                                chance_accuracy=chance,history=history,accuracy=history[-1]['accuracy'],loss=history[-1]['loss'],
                                last_train_loss=loss,training_seconds=training_seconds)
                    results.append(result)
                    path.write_text(json.dumps(dict(config=config,runs=results),indent=2)+'\n')


if __name__=='__main__': main()
