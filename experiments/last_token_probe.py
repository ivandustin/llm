"""Quick controlled multi-query probe of an explicit last-token readout input.

Run from the repo root: python -m experiments.last_token_probe
This experiment leaves the production models unchanged. The duplicate-pool
control matches the last-token variant's parameter count and initialization,
but supplies the pooled vector in both halves of the readout input.
"""
import argparse
import json
import time
from pathlib import Path

import optax
from experiments.benchmark import nnx, np, jnp, jax, count
from experiments.budget_sweep import BatchedCandidate
from experiments.mqar_test import generate
from experiments.speed_benchmark import compiled_training


class LastToken(nnx.Module):
    def __init__(self, seed, duplicate_pool=False, hidden=120, layers=6):
        self.duplicate_pool = duplicate_pool
        self.base = BatchedCandidate(16, 96, 64, hidden, seed, layers).model
        self.base.fn.a = nnx.Linear(128, hidden, rngs=nnx.Rngs(seed + 100000))

    def __call__(self, tokens):
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        def one(x):
            e = self.base.embeds(x)
            # Retain the original prefix pool, including the last token.
            pooled = jnp.prod(e, axis=-3).sum(axis=-2)
            pooled = (pooled - pooled.mean()) / jnp.sqrt(pooled.var() + 1e-5)
            last = e[1, -1]
            last = (last - last.mean()) / jnp.sqrt(last.var() + 1e-5)
            if self.duplicate_pool:
                last = pooled
            h = jax.nn.gelu(self.base.fn.a(jnp.concatenate([pooled, last])))
            for layer in getattr(self.base, 'layers', []):
                h = jax.nn.gelu(layer(h))
            return self.base.fn.b(h)
        return jax.vmap(one)(jnp.stack([positions, tokens], axis=1))


class Probe(nnx.Module):
    def __init__(self, seed, variant, hidden=120, layers=6, last_hidden=None):
        self.base = (TokenInteraction(seed, variant, hidden, layers)
                     if variant in ['previous_pool', 'token_product', 'token_product_gelu', 'global_query', 'token_position_product'] else
                     LastToken(seed, variant == 'duplicate_pool', last_hidden or hidden, layers)
                     if variant != 'baseline' else
                     BatchedCandidate(16, 96, 64, hidden, seed, layers))

    def __call__(self, tokens):
        return jnp.stack([self.base(tokens[:, :end]) for end in [11, 13, 15]], axis=1)


class TokenInteraction(nnx.Module):
    """Query-conditioned pooling (global_query adds 64 trainable parameters).

    global_query includes every token and uses a learned, input-independent
    query initialized to ones. Other variants exclude the final token from
    memory. token_position_product multiplies the last token and its position
    to form the query, reusing the existing embedding tables.
    previous_pool is the no-query control. token_product_last_concat
    appends the last-token embedding to the pooled memory. token_product_gelu
    pools GELU interactions, while token_product_raw_gelu pools both raw and GELU
    interactions. token_product_raw_half replaces the GELU sum with half the
    raw sum. token_product_pre_norm normalizes each interaction across its
    channels before summing. Readout activation and pooled normalization can
    be changed without changing parameter initialization or counts.
    """
    def __init__(self, seed, variant, hidden, layers, context=16, vocab=96,
                 activation='gelu', pooled_norm=True, width=64):
        if activation not in ['gelu', 'sigmoid']:
            raise ValueError('activation must be gelu or sigmoid')
        self.variant = variant
        self.activation = activation
        self.pooled_norm = pooled_norm
        self.base = BatchedCandidate(context, vocab, width, hidden, seed, layers).model
        if variant in ('token_product_raw_gelu', 'token_product_raw_half', 'token_product_raw_sigmoid',
                       'token_product_last_concat'):
            self.base.fn.a = nnx.Linear(2 * width, hidden, rngs=nnx.Rngs(seed + 100000))
        if variant == 'global_query':
            self.query = nnx.Param(jnp.ones(width))
        if variant in ('token_product_norm_query_learned', 'token_product_norm_query_learned_add'):
            self.memory_scale = nnx.Param(jnp.ones(width))
            self.memory_bias = nnx.Param(jnp.zeros(width))
            self.pool_scale = nnx.Param(jnp.ones(width))
            self.pool_bias = nnx.Param(jnp.zeros(width))

    def __call__(self, tokens):
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        def one(x):
            e = self.base.embeds(x)
            memory = jnp.prod(e if self.variant == 'global_query' else e[:, :-1], axis=0)
            if self.variant == 'global_query':
                memory = memory * self.query[...]
            elif self.variant == 'token_position_product':
                memory = memory * (e[0, -1] * e[1, -1])
            elif self.variant in ('token_product_norm_softmax', 'token_product_norm_query', 'token_product_norm_query_learned', 'token_product_norm_query_learned_add'):
                memory = (memory - memory.mean(axis=-1, keepdims=True)) / jnp.sqrt(
                    memory.var(axis=-1, keepdims=True) + 1e-5
                )
                if self.variant in ('token_product_norm_query_learned', 'token_product_norm_query_learned_add'):
                    memory = memory * self.memory_scale[...] + self.memory_bias[...]
                query = jax.nn.softmax(e[1, -1], axis=-1) if self.variant == 'token_product_norm_softmax' else e[1, -1]
                memory = memory * query
            elif self.variant not in ('previous_pool', 'token_product_last_concat'):
                memory = memory * e[1, -1]
            if self.variant == 'token_product_gelu':
                memory = jax.nn.gelu(memory)
            if self.variant == 'token_product_pre_norm':
                memory = (memory - memory.mean(axis=-1, keepdims=True)) / jnp.sqrt(
                    memory.var(axis=-1, keepdims=True) + 1e-5
                )
            pooled = memory.sum(axis=0)
            if self.variant == 'token_product_last_concat':
                pooled = jnp.concatenate([pooled, e[1, -1]])
            elif self.variant == 'token_product_raw_gelu':
                pooled = jnp.concatenate([pooled, jax.nn.gelu(memory).sum(axis=0)])
            elif self.variant in ('token_product_raw_gelu_add', 'token_product_norm_query_learned_add'):
                pooled = pooled + jax.nn.gelu(memory).sum(axis=0)
            elif self.variant == 'token_product_raw_gelu_mul':
                pooled = pooled * jax.nn.gelu(memory).sum(axis=0)
            elif self.variant == 'token_product_raw_sigmoid':
                pooled = jnp.concatenate([pooled, jax.nn.sigmoid(memory).sum(axis=0)])
            elif self.variant == 'token_product_raw_half':
                pooled = jnp.concatenate([pooled, 0.5 * pooled])
            if self.pooled_norm:
                pooled = (pooled - pooled.mean()) / jnp.sqrt(pooled.var() + 1e-5)
                if self.variant in ('token_product_norm_query_learned', 'token_product_norm_query_learned_add'):
                    pooled = pooled * self.pool_scale[...] + self.pool_bias[...]
            activate = jax.nn.gelu if self.activation == 'gelu' else jax.nn.sigmoid
            h = activate(self.base.fn.a(pooled))
            if hasattr(self.base, 'extra'):
                h = activate(self.base.extra(h))
            for layer in getattr(self.base, 'layers', []):
                h = activate(layer(h))
            return self.base.fn.b(h)
        return jax.vmap(one)(jnp.stack([positions, tokens], axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=120)
    parser.add_argument('--last-hidden', type=int)
    parser.add_argument('--output', default='experiments/last-token-probe.json')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'last_token', 'duplicate_pool',
                        'previous_pool', 'token_product', 'token_product_gelu', 'global_query', 'token_position_product'],
                        default=['baseline', 'last_token', 'duplicate_pool'])
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.layers, args.hidden, args.last_hidden or args.hidden) < 1:
        parser.error('counts and widths must be positive')
    evaluation = generate(16384, np.random.default_rng(719002))
    result = dict(config=vars(args) | dict(task='multi', batch=64, lr=.001,
        width=64,
        note='Same training data per seed; final checkpoints; parameter counts recorded per run. Earlier answers are teacher-forced.'), runs=[])
    for seed in range(args.seeds):
        rng = np.random.default_rng(50000 + seed)
        bank = [generate(64, rng) for _ in range(args.steps)]
        xs, ys = jax.device_put(tuple(np.stack(a) for a in zip(*bank)))
        for variant in args.variants:
            started = time.monotonic()
            model = Probe(seed, variant, args.hidden, args.layers, args.last_hidden)
            state, run = compiled_training(model, optax.adam(.001))
            state, losses = run(state, xs, ys)
            jax.block_until_ready((state, losses))
            nnx.update(model, state[0])
            predict = nnx.jit(lambda m, x: m(x).argmax(-1))
            predicted = np.concatenate([np.asarray(predict(model, evaluation[0][i:i+128]))
                                        for i in range(0, len(evaluation[0]), 128)])
            correct = predicted == evaluation[1]
            row = dict(seed=seed, variant=variant, parameters=count(model),
                accuracy=float(correct.mean()), per_query=correct.mean(axis=0).tolist(),
                all_correct=float(correct.all(axis=1).mean()), seconds=time.monotonic()-started)
            result['runs'].append(row)
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
