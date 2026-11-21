"""Constructive lossless pooling check using the local base model embeddings.

This is a hand-designed encoding, not a learned-model performance result.
With d=L, P[i]=unit vector i and E[token]=token * ones(L), pooling
returns the exact vector of token IDs. A second construction assigns all
weights of a wider network to solve queried copying exactly.
"""
import json
from pathlib import Path
from experiments.benchmark import Model, nnx, np, jnp


def main():
    results=[]
    for context in [16,64]:
        vocab=96
        model=Model([context,vocab],vocab,context,32,nnx.Rngs(0))
        model.embeds.embeds[0].embedding[...] = jnp.eye(context)
        model.embeds.embeds[1].embedding[...] = jnp.broadcast_to(jnp.arange(vocab)[:,None],(vocab,context)).astype(jnp.float32)
        tokens=np.random.default_rng(context).integers(0,vocab,(1024,context),dtype=np.int32)
        positions=np.broadcast_to(np.arange(context),tokens.shape)
        embedded=nnx.vmap(model.embeds)(jnp.stack([positions,tokens],axis=1))
        pooled=jnp.sum(jnp.prod(embedded,axis=1),axis=1)
        recovered=np.asarray(pooled).astype(np.int32)
        assert np.array_equal(recovered,tokens)
        result=dict(context=context,embedding_width=context,examples=1024,
                    exact_sequence_recovery=float((recovered==tokens).all(-1).mean()),
                    max_absolute_error=float(np.max(np.abs(np.asarray(pooled)-tokens))))
        results.append(result)
    # Construct an entire queried-copy network. One-hot blocks preserve positions;
    # each hidden unit detects (query == position) AND (token == value).
    from experiments.capabilities import generate
    context,vocab,values=16,96,16
    hidden=(context-1)*values
    width=context*vocab
    model=Model([context,vocab],vocab,width,hidden,nnx.Rngs(0))
    model.embeds.embeds[0].embedding[...] = jnp.repeat(jnp.eye(context),vocab,axis=1)
    model.embeds.embeds[1].embedding[...] = jnp.tile(jnp.eye(vocab),(1,context))
    a=np.zeros((width,hidden),dtype=np.float32)
    b=np.zeros((hidden,vocab),dtype=np.float32)
    for position in range(context-1):
        for token in range(values):
            neuron=position*values+token
            a[position*vocab+token,neuron]=20.
            a[(context-1)*vocab+16+position,neuron]=20.
            b[neuron,token]=10.
    model.fn.a.kernel[...] = jnp.array(a)
    model.fn.a.bias[...] = jnp.full((hidden,),-30.)
    model.fn.b.kernel[...] = jnp.array(b)
    model.fn.b.bias[...] = jnp.zeros((vocab,))
    x,y,_=generate('indexed_copy',context,4096,np.random.default_rng(90000+100*context+1))
    positions=np.broadcast_to(np.arange(context),x.shape)
    prediction=nnx.jit(nnx.vmap(model))(jnp.stack([positions,x],axis=1)).argmax(-1)
    assert np.array_equal(np.asarray(prediction),y)
    from experiments.benchmark import count
    full_network=dict(task='indexed_copy',context=context,vocab=vocab,embedding_width=width,
                      hidden=hidden,parameters=count(model),examples=len(y),accuracy=float((np.asarray(prediction)==y).mean()),
                      note='All weights assigned by construction; no training or parameter matching.')
    output=dict(kind='constructive_hand_designed_encoding_not_learned',
                construction='P[i] = unit_vector(i); E[t] = t * ones(L); sum_i P[i] * E[x_i] = x',
                lossless_pooling_runs=results,constructed_full_network=full_network)
    Path('information-check.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
