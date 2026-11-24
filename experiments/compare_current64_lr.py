"""Compare the complete d64 LR .0003 rerun with the original LR .001 runs.

Run after run_current64_50k --lr .0003 finishes. All milestones are reported;
this does not choose a best checkpoint or replace the original reference files.
"""
import json
from pathlib import Path

from experiments.benchmark import np
from experiments.depth_100k_test import Model as Transformer
from experiments.longer_training_100k import TokenProductModel
from experiments.mqar_test import generate
from experiments.investigate_single_query import score
from experiments.validate_layer_depth import load_weights

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/current64-lr0.0003-50k'
STEPS = [1000, 10000, 30000, 50000]
ARCHES = ['token_product', 'transformer']
TASKS = ['position', 'single', 'multi', 'language']


def main():
    current = json.loads((OUT / 'summary.json').read_text())
    previous = json.loads((ROOT / 'experiments/current64-50k/summary.json').read_text())
    for arch in ARCHES:
        assert all(c['lr'] == .0003 for c in current['configs'][arch])
        assert all(c['lr'] == .001 for c in previous['configs'][arch])
    rows = []
    for row in current['rows']:
        old = next(r for r in previous['rows'] if all(r[k] == row[k] for k in ['architecture', 'task', 'step']))
        rows.append(dict(row, previous_lr_001_accuracy_mean=old['accuracy_mean'],
                         previous_lr_001_accuracy_sd=old['accuracy_sd'],
                         accuracy_delta_percentage_points=100*(row['accuracy_mean']-old['accuracy_mean']),
                         previous_lr_001_perplexity_mean=old['perplexity_mean']))
    lines = [
        'Current d64 models, Adam learning rate .0003, three seeds.',
        'All four tasks; predeclared final milestones 1k/10k/30k/50k.',
        'Mean +/- sample standard deviation; accuracy in percent.',
        'Identical data streams and evaluation samples across models and learning rates.',
        'Continuous model and Adam state; batch64; no schedule or weight decay.',
        'Pooling:100000 retrieval/81153 language parameters; Transformer:98032/94033.',
        'Retrieval:16384 examples. Language:Tiny Shakespeare,80/10/10 split,context32,4096 test windows.',
        'Equal updates are not equal compute. No best-step selection.',
        '',
    ]
    for step in STEPS:
        lines += [f'{step} updates: pooling | Transformer', '']
        for task in TASKS:
            pair = [next(r for r in rows if r['architecture'] == arch and r['task'] == task and r['step'] == step) for arch in ARCHES]
            cells = [f"{100*r['accuracy_mean']:.2f}% +/- {100*r['accuracy_sd']:.2f}" for r in pair]
            lines.append(f'{task} accuracy: ' + ' | '.join(cells))
            if task == 'language':
                lines.append('language perplexity: ' + ' | '.join(f"{r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" for r in pair))
        lines.append('')
    lines += ['50k change from LR .001 to .0003 (accuracy percentage points):', '']
    for row in rows:
        if row['step'] == 50000:
            lines.append(f"{row['architecture']} / {row['task']}: {row['accuracy_delta_percentage_points']:+.2f}")
    # Also use the identical final sample from the single-query investigation.
    x, y = generate(16384, np.random.default_rng(2026112403))
    confirmation = []
    for arch in ARCHES:
        for seed in range(3):
            model = TokenProductModel('single', seed) if arch == 'token_product' else Transformer('single', seed, 'transformer')
            for step in STEPS:
                load_weights(model, OUT / f'{arch}-seed-{seed}/single-step-{step}.npz')
                record = dict(architecture=arch, seed=seed, step=step, metrics=score(model, x, y[:, :1]))
                confirmation.append(record)
                print('single-query confirmation', record, flush=True)
    result = dict(lr=.0003, original_lr=.001, rows=rows,
                  single_query_investigation_test=dict(seed=2026112403, examples=16384, runs=confirmation))
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    lines += [
        '',
        'comparison.json also evaluates every single-query checkpoint on investigation sample seed2026112403.',
        'These are exploratory benchmarks; evaluation samples have been reused across experiments.',
        '',
        'Reproduce from the repository root:',
        '  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.run_current64_50k --lr .0003',
        '  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.compare_current64_lr',
    ]
    report = '\n'.join(lines)+'\n'
    (OUT / 'comparison.txt').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
