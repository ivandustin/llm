"""Quick learned versus fixed sinusoidal positions on the current TF reference."""
import argparse
import json
import time
from pathlib import Path
import optax
from experiments.benchmark import nnx, np, jax, jnp, count
from experiments.depth_100k_test import Model
from experiments.longer_training_100k import training_batch, evaluation_data, TASKS
from experiments.language_depth_test import evaluate
from experiments.speed_benchmark import compiled_training
from experiments.sinusoidal_test import positions

class Sinusoidal(nnx.Module):
    def __call__(self, indices):
        return positions(indices.shape[0], 64)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--output', default='experiments/quick-tf-positions.json')
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 100:
        parser.error('steps must be a positive multiple of 100')
    raw = (Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_text()
    enc = {c: i for i, c in enumerate(sorted(set(raw)))}
    data = np.array([enc[c] for c in raw], np.int32)
    train, val, test = np.split(data, [int(len(data)*.8), int(len(data)*.9)])
    result = dict(config=dict(steps=args.steps, seed=args.seed, width=64, blocks=2,
        heads=4, ff_hidden=200, batch=64, lr=.001, optimizer='Adam', tasks=args.tasks,
        token_scaling='unchanged, no sqrt(d) scaling',
        evaluation='16384 retrieval sequences; 4096 language test windows; final checkpoint'), runs=[])
    for task in args.tasks:
        _, heldout = evaluation_data(task, val, test)
        for kind in ['learned', 'sinusoidal']:
            model = Model(task, args.seed, 'transformer', positions='learned')
            if kind == 'sinusoidal':
                model.base.base.position = Sinusoidal()
            rng = np.random.default_rng(args.seed if task == 'language' else
                10000*args.seed+1601 if task == 'position' else 50000+args.seed)
            state, run = compiled_training(model, optax.adam(.001))
            start = time.perf_counter()
            for _ in range(args.steps // 100):
                pairs = [training_batch(task, rng, train) for _ in range(100)]
                xs, ys = jax.device_put(np.stack([p[0] for p in pairs])), jax.device_put(np.stack([p[1] for p in pairs]))
                state, losses = run(state, xs, ys)
                jax.block_until_ready((state, losses))
            nnx.update(model, state[0])
            row = dict(task=task, positions=kind, parameters=count(model),
                train_seconds_including_compile=time.perf_counter()-start,
                final=evaluate(model, *heldout))
            result['runs'].append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
