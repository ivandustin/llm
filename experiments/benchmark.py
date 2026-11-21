"""Matched-parameter next-character benchmark using the local pooling model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from position_token_pooling import Model


class Transformer(nnx.Module):
    def __init__(self, vocab, context, width, rngs):
        self.token = nnx.Embed(vocab, width, rngs=rngs)
        self.position = nnx.Embed(context, width, rngs=rngs)
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3 * width, rngs=rngs)
        self.proj = nnx.Linear(width, width, rngs=rngs)
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.up = nnx.Linear(width, 4 * width, rngs=rngs)
        self.down = nnx.Linear(4 * width, width, rngs=rngs)
        self.final_norm = nnx.LayerNorm(width, rngs=rngs)
        self.head = nnx.Linear(width, vocab, rngs=rngs)

    def __call__(self, tokens):
        x = self.token(tokens) + self.position(jnp.arange(tokens.shape[-1]))
        z = self.norm1(x)
        batch, length, width = z.shape
        qkv = self.qkv(z).reshape(batch, length, 3, 4, width // 4)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(width // 4)
        causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
        weights = jax.nn.softmax(jnp.where(causal, scores, -1e9), axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
        x = x + self.proj(attended)
        x = x + self.down(jax.nn.gelu(self.up(self.norm2(x))))
        return self.head(self.final_norm(x[:, -1]))


def count(model):
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def logits(model, tokens, cube):
    if cube:
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        return nnx.vmap(model)(jnp.stack([positions, tokens], axis=1))
    return model(tokens)


@nnx.jit(static_argnames=('cube',))
def step(model, optimizer, tokens, targets, cube):
    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(logits(m, tokens, cube), targets).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit(static_argnames=('cube',))
def evaluate(model, tokens, targets, cube):
    pred = logits(model, tokens, cube)
    return (optax.softmax_cross_entropy_with_integer_labels(pred, targets).mean(),
            (pred.argmax(-1) == targets).mean())


def batch(data, starts, context):
    return data[starts[:, None] + np.arange(context)], data[starts + context]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='results.json')
    parser.add_argument('--models', nargs='+', choices=['cube','transformer'], default=['cube','transformer'])
    parser.add_argument('--cube-variant', choices=['original','norm_gelu'], default='original')
    parser.add_argument('--seed-start', type=int, default=0)
    args = parser.parse_args()
    CubeModel=Model
    if args.cube_variant=='norm_gelu':
        from experiments.candidate_model import Model as CubeModel
    raw = (Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_bytes()
    text = raw.decode()
    chars = sorted(set(text))
    encoding = {c: i for i, c in enumerate(chars)}
    data = np.array([encoding[c] for c in text], dtype=np.int32)
    # Contiguous splits; windows never cross split boundaries.
    train, validation, test = np.split(data, [int(len(data)*.8), int(len(data)*.9)])
    vocab, width = len(chars), 32
    target_count = count(Transformer(vocab, args.context, width, nnx.Rngs(0)))
    hidden = min(range(1, 1024), key=lambda n: abs(
        (args.context+vocab)*width + (width+1)*n + (n+1)*vocab - target_count))
    config = vars(args) | dict(vocab=vocab, width=width, hidden=hidden,
        dataset_sha256=hashlib.sha256(raw).hexdigest(), split_lengths=list(map(len, [train, validation, test])),
        devices=[str(d) for d in jax.devices()], jax=jax.__version__)
    print(json.dumps(config), flush=True)
    results = []
    fixed_rng = np.random.default_rng(8675309)
    eval_batches = {name: [batch(part, fixed_rng.integers(0, len(part)-args.context, size=128), args.context)
                         for _ in range(32)] for name, part in [('validation', validation), ('test', test)]}
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for name in args.models:
            cube = name == 'cube'
            model = (CubeModel([args.context, vocab], vocab, width, hidden, nnx.Rngs(seed)) if cube
                     else Transformer(vocab, args.context, width, nnx.Rngs(seed)))
            optimizer = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)
            rng = np.random.default_rng(seed)
            # Compile/update once and report its time separately.
            x, y = batch(train, rng.integers(0, len(train)-args.context, args.batch), args.context)
            compile_start = time.perf_counter()
            first_loss = float(step(model, optimizer, x, y, cube))
            compile_seconds = time.perf_counter() - compile_start
            losses = [first_loss]
            train_start = time.perf_counter()
            for i in range(1, args.steps):
                x, y = batch(train, rng.integers(0, len(train)-args.context, args.batch), args.context)
                losses.append(float(step(model, optimizer, x, y, cube)))
                if (i+1) % 250 == 0:
                    print(f'{name} seed={seed} step={i+1} loss={np.mean(losses[-100:]):.4f}', flush=True)
            train_seconds = time.perf_counter() - train_start
            result = dict(model=name, seed=seed, parameters=count(model), compile_seconds=compile_seconds,
                          train_seconds=train_seconds, final_train_loss=float(np.mean(losses[-100:])))
            for split, batches in eval_batches.items():
                metrics = np.array([[float(v) for v in evaluate(model, x, y, cube)] for x,y in batches]).mean(0)
                result[split] = dict(loss=float(metrics[0]), perplexity=float(np.exp(metrics[0])), accuracy=float(metrics[1]))
            # Autoregressive example at temperature 0.8 with a fixed prompt and seed.
            prompt = 'First Citizen:\n'
            sequence = [encoding[c] for c in ('\n' * args.context + prompt)]
            key = jax.random.key(seed)
            for _ in range(160):
                key, subkey = jax.random.split(key)
                pred = logits(model, jnp.array([sequence[-args.context:]]), cube)[0]
                sequence.append(int(jax.random.categorical(subkey, pred / .8)))
            result['sample'] = ''.join(chars[i] for i in sequence[args.context:])
            results.append(result)
            print(json.dumps(result), flush=True)
            Path(args.output).write_text(json.dumps(dict(config=config, runs=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
