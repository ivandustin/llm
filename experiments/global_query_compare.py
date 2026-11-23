"""Matched final-checkpoint comparison of last-token and global-query pooling."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import optax
from experiments.benchmark import np, jax, nnx, count
from experiments.longer_training_100k import (
    TokenProductModel, TASKS, training_batch, evaluation_data,
)
from experiments.language_depth_test import evaluate
from experiments.speed_benchmark import compiled_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variants', nargs='+',
                        choices=['token_product', 'global_query', 'token_position_product'],
                        default=['token_product', 'global_query'])
    parser.add_argument('--output', default='experiments/global-query-all-quick.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    raw = (Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_bytes()
    text = raw.decode()
    enc = {c: i for i, c in enumerate(sorted(set(text)))}
    data = np.array([enc[c] for c in text], np.int32)
    train, val, test = np.split(data, [int(len(data)*.8), int(len(data)*.9)])
    result = dict(config=dict(steps=args.steps, seed=args.seed, batch=64,
        optimizer='Adam', lr=.001, width=64, hidden=576, hidden_layers=1,
        activation='GELU', pooled_norm='parameter-free epsilon 1e-5',
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        evaluation='final checkpoint; 16384 retrieval sequences, 4096 language windows',
        variants=dict(token_product='last-token query; excludes last position from memory',
                      global_query='learned 64-vector initialized to ones; includes all positions',
                      token_position_product='last token times last position query; excludes last position from memory'),
        selected_variants=args.variants),
        runs=[])
    for task in TASKS:
        _, heldout = evaluation_data(task, val, test)
        for variant in args.variants:
            started = time.monotonic()
            model = TokenProductModel(task, args.seed, variant=variant)
            state, run = compiled_training(model, optax.adam(.001))
            rng = np.random.default_rng(args.seed if task == 'language' else
                10000*args.seed+1601 if task == 'position' else 50000+args.seed)
            for start in range(0, args.steps, 1000):
                pairs = [training_batch(task, rng, train)
                         for _ in range(min(1000, args.steps-start))]
                xs, ys = jax.device_put(tuple(np.stack(p) for p in zip(*pairs)))
                state, losses = run(state, xs, ys)
                jax.block_until_ready((state, losses))
            nnx.update(model, state[0])
            final = evaluate(model, *heldout)
            if task != 'language':
                final.pop('perplexity', None)
            row = dict(task=task, variant=variant, parameters=count(model),
                       final=final, seconds=time.monotonic()-started)
            result['runs'].append(row)
            Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
