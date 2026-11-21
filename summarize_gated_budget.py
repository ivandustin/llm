"""Validate the sigmoid-only gate sweep and compare saved matched controls."""
import json
from pathlib import Path
import numpy as np


def main():
    parts=[json.loads(Path(f'gated-budget-seed-{s}.json').read_text()) for s in range(3)]
    config=parts[0]['config']
    for p in parts:
        for key in ['widths','budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed','hidden_layers','variant']:
            assert p['config'][key]==config[key],key
    assert config['hidden_layers']==1 and config['variant']=='normalized_GELU_sigmoid_query_gate'
    runs=[r for p in parts for r in p['runs']]
    keys=[(r['width'],r['seed']) for r in runs]
    expected={(w,s) for w in config['widths'] for s in range(3)}
    assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'Some runs are incomplete'
    assert all(r['parameters']<=10000 and r['final']['step']==10000 and np.isfinite(r['final']['loss']) for r in runs)
    shallow=json.loads(Path('budget-summary.json').read_text())
    deep=json.loads(Path('depth-budget-summary.json').read_text())
    for control in [shallow,deep]:
        for key in ['budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed','training_seed_formula']:
            assert config[key]==control['config'][key],key
    config=config|dict(seeds=3,seed_start=0,output='gated-budget-results.json',
        execution_note='Independent seeds run concurrently. Saved ungated and transformer controls share the same batches, evaluation and training budget.')
    Path('gated-budget-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    rows=[]
    for width in config['widths']:
        a=next(r for r in shallow['rows'] if r['model']=='candidate' and r['width']==width)
        b=next(r for r in deep['rows'] if r['width']==width)
        rs=[r for r in runs if r['width']==width]
        vals=[100*r['final']['accuracy'] for r in rs]
        rows.append(dict(width=width,hidden=rs[0]['hidden'],parameters=rs[0]['parameters'],
            accuracy_mean=float(np.mean(vals)),accuracy_sd=float(np.std(vals,ddof=1)),
            shallow_accuracy=a['accuracy_mean'],deep_accuracy=b['deep_accuracy']))
    reference=next(r for r in shallow['rows'] if r['model']=='transformer')
    summary=dict(config=config,rows=rows,transformer_reference=reference,frequency_baseline_accuracy=shallow['frequency_baseline_accuracy'])
    Path('gated-budget-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Sigmoid query gate, one hidden stage, <=10,000 total parameters.',
           'Queried copying, context 16, 10,000 Adam updates, three seeds.',
           'Feature branch: normalized pooled vector -> Linear -> GELU.',
           'Gate branch: shared last-token embedding -> Linear -> sigmoid.',
           'Branch product -> output Linear. No second serial hidden layer.',
           '', 'Dims Hidden Parameters   Gated accuracy (%)  Ungated 1 hidden  Ungated 2 hidden']
    for r in rows:
        lines.append(f"{r['width']:4d} {r['hidden']:6d} {r['parameters']:10,d}   {r['accuracy_mean']:6.2f} +/- {r['accuracy_sd']:5.2f}       {r['shallow_accuracy']:6.2f}           {r['deep_accuracy']:6.2f}")
    lines.extend(['',f"Saved one-block transformer: {reference['accuracy_mean']:.2f}%, {reference['parameters']:,} parameters.",
                  f"Frequency shortcut baseline: {100*summary['frequency_baseline_accuracy']:.2f}%."])
    text='\n'.join(lines)+'\n'
    Path('gated-budget-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
