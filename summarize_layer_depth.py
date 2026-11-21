"""Validate 3/4/6-layer sweeps and combine saved 1/2-layer controls."""
import json
from pathlib import Path
import numpy as np

NEW_DEPTHS=[3,4,6]


def main():
    base=json.loads(Path('budget-results.json').read_text())
    old_deep=json.loads(Path('depth-budget-results.json').read_text())
    config=base['config']
    common=['widths','budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed','training_seed_formula']
    for key in common:assert old_deep['config'][key]==config[key],key
    all_runs=[dict(r,hidden_layers=1 if r['model']=='candidate' else None) for r in base['runs']]
    all_runs.extend(dict(r,hidden_layers=2) for r in old_deep['runs'])
    for depth in NEW_DEPTHS:
        new=[]
        for seed in range(3):
            part=json.loads(Path(f'stack-depth-{depth}-seed-{seed}.json').read_text())
            assert part['config']['hidden_layers']==depth
            for key in common:assert part['config'][key]==config[key],key
            assert all(r['model']=='candidate' and r['seed']==seed for r in part['runs'])
            new.extend(dict(r,hidden_layers=depth) for r in part['runs'])
        keys=[(r['width'],r['seed']) for r in new]
        expected={(w,s) for w in config['widths'] for s in range(3)}
        assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected, f'Depth {depth} incomplete'
        all_runs.extend(new)
    assert len(all_runs)==138
    assert all(r['parameters']<=10000 and r['final']['step']==10000 for r in all_runs)
    assert all(np.isfinite(r['final']['loss']) and 0<=r['final']['accuracy']<=1 for r in all_runs)
    config=config|dict(depths=[1,2]+NEW_DEPTHS,new_depths=NEW_DEPTHS,output='layer-depth-results.json',
        execution_note='81 new fits; 57 saved controls reused. Three seed jobs run concurrently, serial depths within each seed; timings not compared.')
    Path('layer-depth-results.json').write_text(json.dumps(dict(config=config,runs=all_runs),indent=2)+'\n')
    rows=[]
    for depth in config['depths']:
        for width in config['widths']:
            rs=[r for r in all_runs if r['model']=='candidate' and r['width']==width and r['hidden_layers']==depth]
            assert len(rs)==3
            vals=[100*r['final']['accuracy'] for r in rs]
            rows.append(dict(hidden_layers=depth,width=width,hidden=rs[0]['hidden'],parameters=rs[0]['parameters'],
                accuracy_mean=float(np.mean(vals)),accuracy_sd=float(np.std(vals,ddof=1)),seed_accuracies=vals))
    best=[max((r for r in rows if r['hidden_layers']==d),key=lambda r:r['accuracy_mean']) for d in config['depths']]
    ref=[r for r in all_runs if r['model']=='transformer']
    vals=[100*r['final']['accuracy'] for r in ref]
    reference=dict(parameters=ref[0]['parameters'],accuracy_mean=float(np.mean(vals)),accuracy_sd=float(np.std(vals,ddof=1)),blocks=1)
    summary=dict(config=config,rows=rows,best_per_depth=best,transformer_reference=reference)
    Path('layer-depth-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Hidden-layer depth sweep: strict <=10,000 total parameters.',
        'Queried copying, context 16, 10,000 Adam updates, three seeds per configuration.',
        'No query gate. Pooled normalization; GELU after every hidden affine layer.',
        '', 'Best observed mean per depth, selected from the same nine embedding widths:',
        'Layers Dims Hidden/unit Parameters  Accuracy (%)        Seed accuracies (%)']
    for r in best:
        lines.append(f"{r['hidden_layers']:6d} {r['width']:4d} {r['hidden']:11d} {r['parameters']:10,d}  {r['accuracy_mean']:6.2f} +/- {r['accuracy_sd']:5.2f}   {r['seed_accuracies']}")
    lines.extend(['',f"Saved one-block transformer: {reference['accuracy_mean']:.2f}% +/- {reference['accuracy_sd']:.2f}, {reference['parameters']:,} parameters.",
                  '', 'Full grid (accuracy means %; SDs and individual seeds are in JSON):',
                  'Dims       1 layer    2 layers   3 layers   4 layers   6 layers'])
    for width in config['widths']:
        means=[next(r['accuracy_mean'] for r in rows if r['width']==width and r['hidden_layers']==d) for d in config['depths']]
        lines.append(f'{width:4d}     '+' '.join(f'{v:10.2f}' for v in means))
    text='\n'.join(lines)+'\n'
    Path('layer-depth-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
