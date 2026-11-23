"""Checkpoint and controlled training diagnostics for single-query TF recall.

Run from the repository root. Writes separate artifacts; reference runs are read-only.
"""
import argparse
import json
from pathlib import Path
import statistics

import optax
from experiments.benchmark import jax, jnp, nnx, np
from experiments.depth_100k_test import Model
from experiments.mqar_test import generate
from experiments.validate_layer_depth import load_weights
from experiments.speed_benchmark import compiled_training
from experiments.readout_experiments import save_weights

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/single-query-investigation'


@nnx.jit
def predict(model, x):
    return model(x)


def logits(model, x):
    return np.concatenate([np.asarray(predict(model, x[i:i+128]))
                           for i in range(0, len(x), 128)])


def score(model, x, y):
    z = logits(model, x)
    return dict(accuracy=(z.argmax(-1) == y).mean(0).tolist(),
                loss=float(optax.softmax_cross_entropy_with_integer_labels(jnp.array(z), jnp.array(y)).mean()))


@nnx.jit
def attention_probe(model, tokens):
    base = model.base.base
    h = base.token(tokens) + base.position(jnp.arange(tokens.shape[1]))
    weights = []
    for block in base.blocks:
        b, length, width = h.shape
        qkv = block.qkv(block.n1(h)).reshape(b, length, 3, 4, width//4)
        q, k = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in [0, 1]]
        mask = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
        weights.append(jax.nn.softmax(jnp.where(mask, q @ k.swapaxes(-1, -2) / jnp.sqrt(width/4), -1e9), -1))
        h = block(h)
    # Per-head attention on the correct value at the first query. Attention is
    # descriptive, not a causal attribution of each head's contribution.
    target = (tokens[:, :10:2] == tokens[:, 10, None]).argmax(-1)*2+1
    query = weights[1][:, :, 10]
    correct = jnp.take_along_axis(query, target[:, None, None], axis=-1)[:, :, 0]
    preceding = jnp.stack([weights[0][:, :, p, p-1] for p in [1, 3, 5, 7, 9]], -1)
    return correct.mean(0), preceding.mean((0, 2))


def checks(model, x, y):
    """Compare optimized path with independent full-prefix forward and gradients."""
    x, y = jnp.array(x[:8]), jnp.array(y[:8, :1])
    def reference(m):
        base = m.base.base
        h = base.token(x[:, :11]) + base.position(jnp.arange(11))
        for block in base.blocks:
            h = block(h)
        return base.head(base.norm(h[:, -1]))[:, None]
    np.testing.assert_allclose(model(x), reference(model), rtol=1e-4, atol=1e-5)
    grad = lambda fn: nnx.grad(lambda m: optax.softmax_cross_entropy_with_integer_labels(fn(m), y).mean())(model)
    for a, b in zip(jax.tree.leaves(grad(lambda m: m(x))), jax.tree.leaves(grad(reference))):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(model(x), model(x[:, :11]), rtol=1e-4, atol=1e-5)
    return 'full-prefix logits/gradients and suffix exclusion passed'


def checkpoints():
    x, y = generate(16384, np.random.default_rng(2026112401))
    counts = (x[:, 1:10:2, None] == np.arange(32, 48)).sum(1)
    bag = counts.argmax(-1) + 32
    report = dict(sample_seed=2026112401, examples=len(x),
                  bag_mode_accuracy=float((bag == y[:, 0]).mean()),
                  bag_optimal_expected_accuracy=float(counts.max(-1).mean()/5), runs=[])
    for seed in range(3):
        for task in ['single', 'multi']:
            model = Model(task, seed, 'transformer')
            for step in [1000, 10000, 50000]:
                load_weights(model, ROOT / f'experiments/current64-50k/transformer-seed-{seed}/{task}-step-{step}.npz')
                row = dict(seed=seed, task=task, step=step, metrics=score(model, x, y[:, :1] if task == 'single' else y))
                attn = attention_probe(model, jnp.array(x[:1024, :11]))
                row['attention'] = dict(layer2_correct_value_by_head=np.asarray(attn[0]).tolist(),
                                        layer1_value_to_preceding_key_by_head=np.asarray(attn[1]).tolist())
                if task == 'single':
                    row['implementation_checks'] = checks(model, x, y)
                    # Each memory is queried for all five keys, eliminating query sampling noise.
                    all_pred = []
                    for k in range(5):
                        changed = x.copy()
                        changed[:, 10] = x[:, 2*k]
                        all_pred.append(logits(model, changed).argmax(-1)[:, 0])
                    all_pred = np.stack(all_pred, 1)
                    row['all_five_query_accuracy'] = float((all_pred == x[:, 1:10:2]).mean())
                    row['same_prediction_for_all_keys'] = float((all_pred == all_pred[:, :1]).all(-1).mean())
                    row['prediction_matches_bag_mode'] = float((all_pred[:, 0] == bag).mean())
                    clipped = np.clip(all_pred-32, 0, 15)
                    selected_counts = np.take_along_axis(counts, clipped, axis=1)
                    valid = (all_pred >= 32) & (all_pred < 48)
                    row['prediction_is_a_bag_mode'] = float(((selected_counts == counts.max(-1, keepdims=True)) & valid).mean())
                print(json.dumps(row), flush=True)
                report['runs'].append(row)
                (OUT / 'checkpoints.json').write_text(json.dumps(report, indent=2)+'\n')


def train(args):
    x, y = generate(4096, np.random.default_rng(2026112402))
    model = Model('single', args.seed, 'transformer')
    if args.warmstart:
        # Same parameter layout; isolate whether a successful multi-query representation
        # transfers to single-query supervision. Adam deliberately starts fresh.
        load_weights(model, ROOT / f'experiments/current64-50k/transformer-seed-{args.seed}/multi-step-10000.npz')
    state, run = compiled_training(model, optax.adam(args.lr))
    rng = np.random.default_rng(50000+args.seed)
    report = dict(seed=args.seed, lr=args.lr, steps=args.steps, warmstart=args.warmstart,
                  evaluation_seed=2026112402, initial=score(model, x, y[:, :1]), history=[])
    name = f'train-seed-{args.seed}-lr-{args.lr}-warm-{args.warmstart}.json'
    for start in range(0, args.steps, 1000):
        pairs = [generate(64, rng) for _ in range(1000)]
        xs = np.stack([p[0] for p in pairs])
        ys = np.stack([p[1][:, :1] for p in pairs])
        state, losses = run(state, xs, ys)
        jax.block_until_ready((state, losses))
        nnx.update(model, state[0])
        row = dict(step=start+1000, metrics=score(model, x, y[:, :1]), train_loss=float(losses[-1]))
        report['history'].append(row)
        (OUT / name).write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(row), flush=True)
    # Separate final sample, never used for the learning curves above.
    final_x, final_y = generate(16384, np.random.default_rng(2026112403))
    report['final_test_seed'] = 2026112403
    report['final_test'] = score(model, final_x, final_y[:, :1])
    report['implementation_checks'] = checks(model, final_x, final_y)
    save_weights(model, OUT / name.replace('.json', '.npz'))
    baseline = Model('single', args.seed, 'transformer')
    load_weights(baseline, ROOT / f'experiments/current64-50k/transformer-seed-{args.seed}/single-step-10000.npz')
    report['reference_lr_001_10k_same_test'] = score(baseline, final_x, final_y[:, :1])
    (OUT / name).write_text(json.dumps(report, indent=2)+'\n')
    print('FINAL', json.dumps(report['final_test']), flush=True)


def attention():
    x, y = generate(4096, np.random.default_rng(2026112401))
    rows = []
    for seed in range(3):
        for task in ['single', 'multi']:
            model = Model('single', seed, 'transformer')
            load_weights(model, ROOT / f'experiments/current64-50k/transformer-seed-{seed}/{task}-step-50000.npz')
            a, b = attention_probe(model, jnp.array(x[:, :11]))
            row = dict(seed=seed, trained_task=task, single_query_score=score(model, x, y[:, :1]),
                       layer2_correct_value_by_head=np.asarray(a).tolist(),
                       layer1_value_to_preceding_key_by_head=np.asarray(b).tolist())
            z = logits(model, x).argmax(-1)[:, 0]
            counts = (x[:, 1:10:2, None] == np.arange(32, 48)).sum(1)
            selected = counts[np.arange(len(x)), np.clip(z-32, 0, 15)]
            row['prediction_is_a_bag_mode'] = float(((selected == counts.max(-1)) & (z >= 32) & (z < 48)).mean())
            rows.append(row)
            print(json.dumps(row), flush=True)
            (OUT / 'attention.json').write_text(json.dumps(rows, indent=2)+'\n')


def summarize():
    runs = [json.loads((OUT / f'train-seed-{s}-lr-0.0003-warm-False.json').read_text()) for s in range(3)]
    for run in runs:
        assert run['steps'] == 10000 and run['history'][-1]['step'] == 10000
        assert 'final_test' in run and run['implementation_checks']
    result = dict(seeds=[0, 1, 2], steps=10000, final_test_seed=2026112403,
                  final_test_examples=16384, trained_from_scratch=True,
                  tuning_note='Exploratory: lower LR selected after seed-0 diagnostic; seeds 1 and 2 are replication. Final test is separate from diagnostic curves.',
                  results={})
    for name, field in [('lr_0003', 'final_test'), ('reference_lr_001', 'reference_lr_001_10k_same_test')]:
        values = [r[field]['accuracy'][0] for r in runs]
        result['results'][name] = dict(accuracy_by_seed=values, accuracy_mean=statistics.mean(values),
                                     accuracy_sd=statistics.stdev(values),
                                     loss_by_seed=[r[field]['loss'] for r in runs])
    probes = []
    x, y = generate(4096, np.random.default_rng(2026112401))
    for seed in range(3):
        model = Model('single', seed, 'transformer')
        load_weights(model, OUT / f'train-seed-{seed}-lr-0.0003-warm-False.npz')
        a, b = attention_probe(model, jnp.array(x[:, :11]))
        preds = []
        for k in range(5):
            changed = x.copy()
            changed[:, 10] = x[:, 2*k]
            preds.append(logits(model, changed).argmax(-1)[:, 0])
        preds = np.stack(preds, 1)
        probes.append(dict(seed=seed, all_five_query_accuracy=float((preds == x[:, 1:10:2]).mean()),
                           same_prediction_for_all_keys=float((preds == preds[:, :1]).all(-1).mean()),
                           layer2_correct_value_by_head=np.asarray(a).tolist(),
                           layer1_value_to_preceding_key_by_head=np.asarray(b).tolist()))
    result['lower_lr_probes'] = probes
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = [
        'Single-query Transformer investigation (2026-11-24)',
        '',
        'Two-block sinusoidal Transformer: d64, four heads, FF200, 98032 parameters.',
        'Adam, batch64, 10000 updates; same initial seed and training stream as reference.',
        'Only learning rate changed. No schedule, weight decay, architecture or generator changes.',
        'Three seeds; means +/- sample SD. Final test: 16384 sequences, seed2026112403.',
        'Exploratory LR selection used seed0 diagnostic curves; seeds1/2 replicate the finding.',
        'The final test sample is separate from the 4096-example diagnostic curves.',
        '',
    ]
    for name, values in result['results'].items():
        lines.append(f"{name}: {100*values['accuracy_mean']:.4f}% +/- {100*values['accuracy_sd']:.4f}; seeds={values['accuracy_by_seed']}")
    lines += [
        '',
        'Checkpoint diagnosis (checkpoints.json and attention.json):',
        '- Independent full-prefix logits and gradients match the optimized single-query path.',
        '- Removing all tokens after the first query leaves predictions unchanged.',
        '- Failed 50k models predict the same answer for all five queried keys in 91-92% of memories.',
        '- Failed models select a most-frequent stored value in 96-98% of the attention-probe examples.',
        '- A value-frequency-only oracle has approximately 30.8% expected accuracy on the checkpoint sample.',
        '- Multi-trained models solve the first query without previous answers; no future answer is needed.',
        '- Attention is consistent with first-layer key/value binding and second-layer retrieval in successful models.',
        '  Attention weights are descriptive evidence, not a causal intervention.',
        '- Seed0 multi10k -> single supervision with fresh Adam .001 remains at 100% after1000 updates.',
        '- Seed0 LR .003 remains at24.02% after10k on the diagnostic sample.',
        '',
        'Conclusion: the .001 result is an optimization/shortcut failure under that training regime.',
        'LR .0003 supplies a successful from-scratch counterexample with the same architecture and data.',
        'This does not isolate why multi-query training escapes: label density, query positions, and contexts differ.',
        'Reference .001 tables are preserved. A broader tuned comparison should give both architectures equal tuning budgets.',
        '',
        'Reproduce from repository root:',
        '  python -m experiments.investigate_single_query',
        '  python -m experiments.investigate_single_query --attention',
        '  python -m experiments.investigate_single_query --train --lr .0003 --seed 0',
        '  python -m experiments.investigate_single_query --train --lr .0003 --seed 1',
        '  python -m experiments.investigate_single_query --train --lr .0003 --seed 2',
        '  python -m experiments.investigate_single_query --train --lr .003 --seed 0',
        '  python -m experiments.investigate_single_query --train --warmstart --steps 1000',
        '  python -m experiments.investigate_single_query --summarize',
        'Execution used OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 and CPU affinity subsets.',
        'Checkpoints contain weights only; Adam state was continuous in memory during each run.',
    ]
    (OUT / 'summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--warmstart', action='store_true')
    args = parser.parse_args()
    if args.steps < 1000 or args.steps % 1000:
        parser.error('steps must be a positive multiple of 1000')
    OUT.mkdir(exist_ok=True)
    if args.train:
        train(args)
    elif args.attention:
        attention()
    elif args.summarize:
        summarize()
    else:
        checkpoints()
