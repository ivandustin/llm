"""Validate the added-layer sweep and compare with saved matched-budget runs."""
import json
from pathlib import Path
import numpy as np


def main():
    parts=[json.loads(Path(f'depth-budget-seed-{s}.json').read_text()) for s in range(3)]
    config=parts[0]['config']
    for p in parts:
        for key in ['widths','budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed','hidden_layers']:
            assert p['config'][key]==config[key],key
    assert config['hidden_layers']==2
    runs=[r for p in parts for r in p['runs']]
    keys=[(r['width'],r['seed']) for r in runs]
    expected={(w,s) for w in config['widths'] for s in range(3)}
    assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'Some runs are incomplete'
    assert all(r['model']=='candidate' and r['parameters']<=10000 and r['final']['step']==10000 for r in runs)
    assert all(np.isfinite(r['final']['loss']) for r in runs)
    previous=json.loads(Path('budget-summary.json').read_text())
    for key in ['budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed','training_seed_formula']:
        assert config[key]==previous['config'][key],key
    config=config|dict(seeds=3,seed_start=0,output='depth-budget-results.json',
        execution_note='Seeds run concurrently. One-hidden-layer and transformer controls reused from budget-results.json; identical task, batches, evaluation and optimizer.')
    Path('depth-budget-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    rows=[]
    for width in config['widths']:
        base=next(r for r in previous['rows'] if r['model']=='candidate' and r['width']==width)
        rs=[r for r in runs if r['width']==width]
        vals=[100*r['final']['accuracy'] for r in rs]
        rows.append(dict(width=width,shallow_hidden=base['hidden'],deep_hidden=rs[0]['hidden'],
            shallow_parameters=base['parameters'],deep_parameters=rs[0]['parameters'],
            shallow_accuracy=base['accuracy_mean'],shallow_sd=base['accuracy_sd'],
            deep_accuracy=float(np.mean(vals)),deep_sd=float(np.std(vals,ddof=1))))
    reference=next(r for r in previous['rows'] if r['model']=='transformer')
    summary=dict(config=config,rows=rows,transformer_reference=reference,
                 frequency_baseline_accuracy=previous['frequency_baseline_accuracy'])
    Path('depth-budget-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Adding one hidden layer at <=10,000 total parameters.',
           'Queried copying, context 16, 10,000 Adam updates, three seeds.',
           'Both variants normalize pooled features and use GELU in every hidden layer.',
           '', 'Dims  Shallow hidden  Deep hidden x2  Deep params  Shallow accuracy  Deep accuracy']
    for r in rows:
        lines.append(f"{r['width']:4d} {r['shallow_hidden']:15d} {r['deep_hidden']:15d} {r['deep_parameters']:12,d}  {r['shallow_accuracy']:5.2f} +/- {r['shallow_sd']:4.2f}   {r['deep_accuracy']:5.2f} +/- {r['deep_sd']:4.2f}")
    lines.extend(['',f"Saved one-block transformer reference: {reference['accuracy_mean']:.2f}% +/- {reference['accuracy_sd']:.2f}, {reference['parameters']:,} parameters.",
                  f"Frequency shortcut baseline: {100*previous['frequency_baseline_accuracy']:.2f}%."])
    text='\n'.join(lines)+'\n'
    Path('depth-budget-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
