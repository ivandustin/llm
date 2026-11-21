"""Recompute shortcut baselines and summarize completed capability runs."""
import json
from pathlib import Path
import numpy as np
from experiments.capabilities import TASKS, generate


def baseline(task, context):
    index = TASKS.index(task)
    x,y,chance = generate(task, context, 4096, np.random.default_rng(90000+100*context+index))
    if task == 'relative_order':
        pred = np.zeros(len(y), dtype=int)
    else:
        if task in ['indexed_copy', 'majority']:
            candidates = x[:,:-1]
        elif task == 'associative_recall':
            candidates = x[:,1:2*((context-1)//2):2]
        else:
            candidates = x
        pred = np.stack([np.bincount(row, minlength=96) for row in candidates]).argmax(-1)
    return dict(task=task, context=context, chance_accuracy=chance,
                frequency_baseline_accuracy=float((pred == y).mean()))


def main():
    source = Path('capabilities-results.json')
    r = json.loads(source.read_text())
    baseline_rows = [baseline(task,c) for c in r['config']['contexts'] for task in TASKS]
    Path('capabilities-baselines.json').write_text(json.dumps(baseline_rows, indent=2)+'\n')
    lines = ['Capability accuracy: mean +/- sample standard deviation across seeds.',
             'Context lengths are trained independently; this is not length extrapolation.',
             'Frequency baseline guesses the modal candidate token (ties: smallest ID).',
             'For relative order the baseline is constant class 0.', '',
             'Task                  Context  Cube (%)         Transformer (%)  Frequency (%)']
    for c in r['config']['contexts']:
        for task in TASKS:
            fields = []
            for name in ['cube','transformer']:
                xs = [100*x['accuracy'] for x in r['runs'] if x['model']==name and x['task']==task and x['context']==c]
                fields.append(f'{np.mean(xs):6.2f} +/- {np.std(xs,ddof=1):4.2f}' if len(xs)==3 else f'pending ({len(xs)}/3)')
            b = next(x for x in baseline_rows if x['task']==task and x['context']==c)
            lines.append(f"{task:22s}{c:7d}  {fields[0]:18s} {fields[1]:18s} {100*b['frequency_baseline_accuracy']:6.2f}")
    for filename in ['capabilities-longer-training.json', 'capabilities-retrieval-extended.json', 'capabilities-wider.json']:
        path = Path(filename)
        if not path.exists(): continue
        extra = json.loads(path.read_text())
        lines.extend(['',filename, json.dumps(extra['config'])])
        for name in ['cube','transformer']:
            xs=[100*x['accuracy'] for x in extra['runs'] if x['model']==name]
            if xs:
                lines.append(f'{name}: {np.mean(xs):.2f}% (n={len(xs)}; individual={xs})')
    path=Path('capabilities-one-block.json')
    if path.exists():
        control=json.loads(path.read_text())
        lines.extend(['','One-block transformer, parameter-matched queried copying:'])
        for context in [16,64]:
            xs=[100*x['accuracy'] for x in control['runs'] if x['context']==context]
            lines.append(f'Context {context}: {np.mean(xs):.2f}% +/- {np.std(xs,ddof=1):.2f} (n={len(xs)})')
    text='\n'.join(lines)+'\n'
    Path('capabilities-summary.txt').write_text(text)
    print(text)


if __name__ == '__main__': main()
