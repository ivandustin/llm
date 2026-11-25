"""Early matched-budget comparison of residual and standard learned-norm pooling."""
import json
from pathlib import Path

import optax
from experiments.benchmark import nnx, np, jax, jnp, count
from experiments.longer_training_100k import TokenProductModel, training_batch, evaluation_data
from experiments.language_depth_test import evaluate
from experiments.speed_benchmark import compiled_training
from position_token_pooling.base import Embeds


class ResidualPool(nnx.Module):
    def __init__(self, task, seed, hidden, width=256, blocks=2):
        self.task = task
        vocab, context = (65, 32) if task == 'language' else (96, 16)
        rng = nnx.Rngs(seed)
        self.embeds = Embeds(width, [context, vocab], rng)
        self.memory_norm = nnx.LayerNorm(width, epsilon=1e-5, rngs=rng)
        self.norm1 = nnx.LayerNorm(width, epsilon=1e-5, rngs=rng)
        if blocks == 2:
            self.norm2 = nnx.LayerNorm(width, epsilon=1e-5, rngs=rng)
        self.final_norm = nnx.LayerNorm(width, epsilon=1e-5, rngs=rng)
        self.a1, self.b1 = nnx.Linear(width, hidden, rngs=rng), nnx.Linear(hidden, width, rngs=rng)
        if blocks == 2:
            self.a2, self.b2 = nnx.Linear(width, hidden, rngs=rng), nnx.Linear(hidden, width, rngs=rng)
        self.head = nnx.Linear(width, vocab, rngs=rng)

    def prefix(self, tokens):
        pos = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        def one(p, t):
            e = self.embeds(jnp.stack([p, t]))
            x = (self.memory_norm(e[0, :-1] * e[1, :-1]) * e[1, -1]).sum(axis=0)
            x = x + self.b1(jax.nn.gelu(self.a1(self.norm1(x))))
            if hasattr(self, 'a2'):
                x = x + self.b2(jax.nn.gelu(self.a2(self.norm2(x))))
            return self.head(self.final_norm(x))
        return jax.vmap(one)(pos, tokens)

    def __call__(self, tokens):
        if self.task in ('single', 'multi'):
            return jnp.stack([self.prefix(tokens[:, :end]) for end in ([11] if self.task == 'single' else [11, 13, 15])], axis=1)
        return self.prefix(tokens)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--blocks', type=int, choices=[1, 2], default=2)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--skip-reference', action='store_true')
    p.add_argument('--tasks', nargs='+', default=['position', 'single', 'multi', 'language'])
    p.add_argument('--output', default='experiments/residual-pool-quick.json')
    args = p.parse_args()
    assert args.steps > 0 and args.steps % 1000 == 0
    raw = Path('data/shakespeare.txt').read_text()
    chars = {c: i for i, c in enumerate(sorted(set(raw)))}
    data = np.array([chars[c] for c in raw], np.int32)
    train, val, test = np.split(data, [int(len(data)*.8), int(len(data)*.9)])
    results = []
    for task in args.tasks:
        base = count(ResidualPool(task, args.seed, 1, width=args.width, blocks=args.blocks))
        hidden = 1 + (100000 - base) // (args.blocks * (2 * args.width + 1))
        for name in (['residual'] if args.skip_reference else ['reference', 'residual']):
            model = (TokenProductModel(task, args.seed, variant='token_product_norm_query_learned', width=256, hidden=230 if task == 'language' else 198)
                     if name == 'reference' else ResidualPool(task, args.seed, hidden, width=args.width, blocks=args.blocks))
            params = count(model)
            assert params <= 100000
            rng = np.random.default_rng(args.seed if task == 'language' else 10000*args.seed+1601 if task == 'position' else 50000+args.seed)
            _, heldout = evaluation_data(task, val, test)
            state, run = compiled_training(model, optax.adam(.0003))
            for start in range(0, args.steps, 1000):
                pairs = [training_batch(task, rng, train) for _ in range(1000)]
                xs = jax.device_put(np.stack([p[0] for p in pairs]))
                ys = jax.device_put(np.stack([p[1] for p in pairs]))
                state, loss = run(state, xs, ys)
                jax.block_until_ready((state, loss))
                print(task, name, start + 1000, float(loss[-1]), flush=True)
            nnx.update(model, state[0])
            result = dict(task=task, model=name, width=256 if name == 'reference' else args.width, blocks=args.blocks if name == 'residual' else None, seed=args.seed, steps=args.steps, hidden=230 if name == 'reference' and task == 'language' else 198 if name == 'reference' else hidden, parameters=params, final=evaluate(model, *heldout))
            results.append(result)
            Path(args.output).write_text(json.dumps({'lr': .0003, 'batch': 64, 'results': results}, indent=2) + '\n')
            print(result, flush=True)


if __name__ == '__main__':
    main()
