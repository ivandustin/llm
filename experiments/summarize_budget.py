"""Merge and validate the strict-budget width sweep."""
import json
from pathlib import Path
import numpy as np
from experiments.capabilities import generate


def main():
    parts=[json.loads(Path(f'budget-seed-{s}.json').read_text()) for s in range(3)]
    config=parts[0]['config']
    for p in parts:
        for key in ['widths','budget','steps','context','vocab','batch','lr','test_examples','evaluation_seed']:
            assert p['config'][key]==config[key],key
    runs=[r for p in parts for r in p['runs']]
    keys=[(r['model'],r['width'],r['seed']) for r in runs]
    expected={('candidate',w,s) for w in config['widths'] for s in range(3)}|{('transformer',16,s) for s in range(3)}
    assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'Some runs are incomplete'
    assert all(r['parameters']<=config['budget'] and r['final']['step']==config['steps'] for r in runs)
    assert all(np.isfinite(r['final']['loss']) and 0<=r['final']['accuracy']<=1 for r in runs)
    config=config|dict(seeds=3,seed_start=0,output='budget-results.json',execution_note='Independent seeds ran concurrently; no timing comparison.')
    Path('budget-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    x,y,_=generate('indexed_copy',config['context'],4096,np.random.default_rng(config['evaluation_seed']))
    mode=np.stack([np.bincount(row,minlength=96) for row in x[:,:-1]]).argmax(-1)
    frequency=float((mode==y).mean())
    rows=[]
    for name,width in [('candidate',w) for w in config['widths']]+[('transformer',16)]:
        rs=[r for r in runs if r['model']==name and r['width']==width]
        a=[100*r['final']['accuracy'] for r in rs]
        rows.append(dict(model=name,width=width,hidden=rs[0]['hidden'],parameters=rs[0]['parameters'],
                         accuracy_mean=float(np.mean(a)),accuracy_sd=float(np.std(a,ddof=1))))
    summary=dict(config=config,frequency_baseline_accuracy=frequency,rows=rows)
    Path('budget-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Strict 10,000-total-parameter budget; queried copying, context 16.',
           'Adam 0.001, batch 64, 10,000 updates; three seeds; mean +/- sample SD.',
           f'Frequency shortcut baseline: {100*frequency:.2f}%. Uniform valid-label chance: 6.25%.',
           'Candidate uses parameter-free normalization + GELU; transformer uses one block.',
           '', 'Model          Dims  Hidden  Parameters  Accuracy (%)']
    for r in rows:
        lines.append(f"{r['model']:14s} {r['width']:4d} {r['hidden']:7d} {r['parameters']:11,d}  {r['accuracy_mean']:6.2f} +/- {r['accuracy_sd']:.2f}")
    text='\n'.join(lines)+'\n'
    Path('budget-summary.txt').write_text(text)
    print(text)


if __name__=='__main__': main()
