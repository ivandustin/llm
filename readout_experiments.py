"""Separate lossless encoding, readout learning, and end-to-end optimization.

The local pooling model supplies embeddings and the original two affine layers.
Fixed variants freeze a task-lossless block-one-hot encoding. Other variants
learn embeddings from the start, or unfreeze after successful readout learning.
"""
import argparse
import json
from pathlib import Path
import time

from benchmark import Model, count, nnx, np, jnp, jax
from capabilities import generate
import optax

VARIANTS=['fixed_sigmoid','fixed_norm','fixed_gelu','fixed_deep','lossless_joint','random_joint','random_norm','random_gelu','random_norm_gelu']


def set_lossless(embeds, context):
    width=(context-1)*17
    p=np.zeros((context,width),np.float32)
    e=np.zeros((96,width),np.float32)
    for position in range(context-1):
        p[position,position*16:(position+1)*16]=1
        e[np.arange(16),position*16+np.arange(16)]=1
    p[-1,(context-1)*16:]=1
    e[16+np.arange(context-1),(context-1)*16+np.arange(context-1)]=1
    embeds.embeds[0].embedding[...] = jnp.array(p)
    embeds.embeds[1].embedding[...] = jnp.array(e)


def pool(embeds, tokens):
    positions=jnp.broadcast_to(jnp.arange(tokens.shape[-1]),tokens.shape)
    e=nnx.vmap(embeds)(jnp.stack([positions,tokens],axis=1))
    return jnp.prod(e,axis=1).sum(axis=1)


class Head(nnx.Module):
    def __init__(self, fn, variant, hidden, rngs):
        self.fn=fn
        self.variant=variant
        self.extra=nnx.Linear(hidden,hidden,rngs=rngs) if variant=='fixed_deep' else None

    def prepare(self,z):
        if 'norm' in self.variant:
            z=(z-z.mean(-1,keepdims=True))/jnp.sqrt(z.var(-1,keepdims=True)+1e-5)
        return z

    def __call__(self,z):
        h=self.fn.a(self.prepare(z))
        h=jax.nn.gelu(h) if 'gelu' in self.variant else jax.nn.sigmoid(h)
        if self.extra is not None:
            h=jax.nn.sigmoid(self.extra(h))
        return self.fn.b(h)


class Joint(nnx.Module):
    def __init__(self, embeds, head):
        self.embeds=embeds
        self.head=head

    def __call__(self,tokens):
        return self.head(pool(self.embeds,tokens))


@nnx.jit
def learn(model, optimizer, x, y):
    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(m(x),y).mean()
    loss,grads=nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model,grads)
    return loss


@nnx.jit
def score(model,x,y):
    pred=model(x)
    return optax.softmax_cross_entropy_with_integer_labels(pred,y).mean(),(pred.argmax(-1)==y).mean()


@nnx.jit
def activation_stats(head,z):
    pre=head.fn.a(head.prepare(z))
    s=jax.nn.sigmoid(pre)
    return jnp.array([pre.mean(),pre.std(),(jnp.abs(pre)>4).mean(),(s*(1-s)).mean()])


def metrics(model,x,y):
    vals=np.mean([[float(v) for v in score(model,x[i:i+128],y[i:i+128])]
                  for i in range(0,len(y),128)],axis=0)
    return dict(loss=float(vals[0]),accuracy=float(vals[1]))


def stats(head,z):
    vals=np.asarray(activation_stats(head,z))
    return dict(zip(['preactivation_mean','preactivation_std','fraction_abs_over_4','mean_sigmoid_derivative'],map(float,vals)))


def save_weights(model,path):
    state=nnx.to_flat_state(nnx.state(model,nnx.Param))
    np.savez_compressed(path,**{'/'.join(map(str,k)):np.asarray(v.get_value()) for k,v in state})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--context',type=int,default=16)
    p.add_argument('--width',type=int)
    p.add_argument('--hidden',type=int)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--output',default='readout-results.json')
    p.add_argument('--unfreeze-steps',type=int,default=3000)
    args=p.parse_args()
    if args.context<4 or args.context>64: p.error('context must be between 4 and 64')
    context=args.context
    width=args.width or (context-1)*17
    if width != (context-1)*17 and any(not v.startswith('random_') for v in args.variants):
        p.error('custom width is supported only for random-embedding variants')
    baseline_hidden=args.hidden or (context-1)*16
    baseline_head=(width+1)*baseline_hidden+(baseline_hidden+1)*96
    deep_hidden=min(range(1,2048),key=lambda h:abs((width+1)*h+(h+1)*h+(h+1)*96-baseline_head))
    config=vars(args)|dict(width=width,vocab=96,batch=64,lr=.001,unfreeze_lr=.0001,
        baseline_hidden=baseline_hidden,deep_hidden=deep_hidden,test_examples=4096,
        stats_note='Sigmoid derivative is a reference diagnostic; GELU variants do not use sigmoid at that layer.',
        evaluation_note='Different RNG from previous capability tests; IID support is not explicitly partitioned.')
    print(json.dumps(config),flush=True)
    output=Path(args.output)
    artifact_dir=output.with_suffix('')
    artifact_dir.mkdir(exist_ok=True)
    test_x,test_y,_=generate('indexed_copy',context,4096,np.random.default_rng(700000+context))
    runs=[]
    pooled=nnx.jit(pool)
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for variant in args.variants:
            hidden=deep_hidden if variant=='fixed_deep' else baseline_hidden
            rngs=nnx.Rngs(seed)
            base=Model([context,96],96,width,hidden,rngs)
            if not variant.startswith('random_'): set_lossless(base.embeds,context)
            head=Head(base.fn,variant,hidden,rngs)
            joint=Joint(base.embeds,head)
            frozen=variant.startswith('fixed_')
            model=head if frozen else joint
            z_test=pooled(joint.embeds,test_x)
            if not variant.startswith('random_'):
                z=np.asarray(z_test)
                decoded=np.concatenate([z[:,:16*(context-1)].reshape(-1,context-1,16).argmax(-1),
                                        (16+z[:,16*(context-1):].argmax(-1))[:,None]],axis=1)
                assert np.array_equal(decoded,test_x),'Encoding is not lossless'
            fixed_before=[np.asarray(e.embedding).copy() for e in joint.embeds.embeds]
            initial_stats=stats(head,z_test)
            initial=metrics(model,z_test if frozen else test_x,test_y)
            optimizer=nnx.Optimizer(model,optax.adam(.001),wrt=nnx.Param)
            rng=np.random.default_rng(10000*seed+100*context+1)
            history=[]
            start=time.perf_counter()
            for i in range(args.steps):
                x,y,_=generate('indexed_copy',context,64,rng)
                loss=float(learn(model,optimizer,pooled(joint.embeds,x) if frozen else x,y))
                if (i+1) in sorted(set([min(500,args.steps),min(1500,args.steps),min(3000,args.steps),args.steps])):
                    point=metrics(model,z_test if frozen else test_x,test_y)|dict(step=i+1,train_loss=loss)
                    history.append(point)
                    print(f'{variant} seed={seed} step={i+1} accuracy={point["accuracy"]:.4f}',flush=True)
            seconds=time.perf_counter()-start
            if frozen:
                assert all(np.array_equal(old,np.asarray(e.embedding)) for old,e in zip(fixed_before,joint.embeds.embeds))
            result=dict(variant=variant,seed=seed,hidden=hidden,parameters=count(joint),trainable_parameters=count(model),
                        initial=initial,initial_stats=initial_stats,final_stats=stats(head,pooled(joint.embeds,test_x)),
                        history=history,final=history[-1],seconds=seconds,embedding_frozen=frozen)
            save_weights(joint,artifact_dir/f'{variant}-seed-{seed}.npz')
            # Paired continuation after baseline success: same initial weights,
            # fresh identical batches, optimizer reset in both branches.
            if frozen and result['final']['accuracy']>=.99 and args.unfreeze_steps:
                continuation=[]
                for unfreeze in [False,True]:
                    continued=nnx.clone(joint)
                    m=continued if unfreeze else continued.head
                    opt=nnx.Optimizer(m,optax.adam(.0001),wrt=nnx.Param)
                    rng2=np.random.default_rng(800000+seed)
                    z=pooled(continued.embeds,test_x)
                    for i in range(args.unfreeze_steps):
                        x,y,_=generate('indexed_copy',context,64,rng2)
                        float(learn(m,opt,x if unfreeze else pooled(continued.embeds,x),y))
                    continuation.append(dict(unfreeze=unfreeze,steps=args.unfreeze_steps,
                        final=metrics(m,test_x if unfreeze else z,test_y)))
                    save_weights(continued,artifact_dir/f'{variant}-seed-{seed}-unfreeze-{unfreeze}.npz')
                result['continuation']=continuation
            runs.append(result)
            output.write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')


if __name__=='__main__':main()
