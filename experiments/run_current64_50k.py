"""Run the current d64 references and summarize all predeclared checkpoints."""
import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/current64-50k'
STEPS = [1000, 10000, 30000, 50000]
ARCHES = ['token_product', 'transformer']
LR = .001


def run_seed(seed):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    cpus = sorted(os.sched_getaffinity(0))
    assigned = cpus[seed::3]
    for arch in ARCHES:
        prefix = OUT / arch
        command = ['taskset', '-c', ','.join(map(str, assigned)), sys.executable,
                   '-m', 'experiments.longer_training_100k', '--architecture', arch,
                   '--positions', 'sinusoidal', '--width', '64', '--steps', '50000',
                   '--lr', str(LR),
                   '--checkpoints', *map(str, STEPS), '--seed', str(seed),
                   '--output-prefix', str(prefix)]
        with (OUT / f'{arch}-seed-{seed}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def summarize():
    rows = []
    configs = {}
    for arch in ARCHES:
        parts = [json.loads((OUT / f'{arch}-seed-{s}.json').read_text()) for s in range(3)]
        configs[arch] = [p['config'] for p in parts]
        for part in parts:
            assert part['config']['lr'] == LR
            assert len(part['runs']) == 4
            for run in part['runs']:
                assert run['complete'] and run['updates_completed'] == 50000
                assert [p['step'] for p in run['checkpoints']] == STEPS
        for task in ['position', 'single', 'multi', 'language']:
            for step in STEPS:
                values = [next(c['final'] for c in next(r for r in p['runs'] if r['task'] == task)['checkpoints'] if c['step'] == step) for p in parts]
                row = dict(architecture=arch, task=task, step=step)
                for metric in ['accuracy', 'loss', 'perplexity']:
                    vals = [v[metric] for v in values]
                    row[metric + '_mean'] = statistics.mean(vals)
                    row[metric + '_sd'] = statistics.stdev(vals)
                rows.append(row)
    (OUT / 'summary.json').write_text(json.dumps(dict(configs=configs, rows=rows), indent=2) + '\n')
    lines = ['Current d64 references; three seeds, mean +/- sample standard deviation.',
             'Pooling: 100000 retrieval / 81153 language parameters; Transformer: 98032 / 94033.',
             f'Adam {LR}, batch 64, continuous state; identical task data streams and fixed evaluation samples.',
             'Equal updates do not imply equal compute. No best-checkpoint selection.', '']
    for row in rows:
        line = f"{row['architecture']} / {row['task']} / {row['step']}: {100*row['accuracy_mean']:.2f}% +/- {100*row['accuracy_sd']:.2f}"
        if row['task'] == 'language':
            line += f"; perplexity {row['perplexity_mean']:.2f} +/- {row['perplexity_sd']:.2f}"
        lines.append(line)
    report = '\n'.join(lines) + '\n'
    (OUT / 'summary.txt').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.lr < float('inf'):
        parser.error('learning rate must be finite and positive')
    LR = args.lr
    OUT = args.output.resolve() if args.output else ROOT / ('experiments/current64-50k' if LR == .001 else f'experiments/current64-lr{LR}-50k')
    OUT.mkdir(exist_ok=True)
    if not args.summarize_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run_seed, range(3)))
    summarize()
