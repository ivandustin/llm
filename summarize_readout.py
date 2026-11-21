"""Merge the three independent readout experiments and summarize controls."""
import json
from pathlib import Path
import numpy as np
from readout_experiments import VARIANTS


def main():
    parts=[json.loads(Path(f'readout-seed-{s}.json').read_text()) for s in range(3)]
    runs=[r for p in parts for r in p['runs']]
    keys=[(r['variant'],r['seed']) for r in runs]
    expected={(v,s) for v in VARIANTS for s in range(3)}
    assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'Some runs are incomplete'
    assert all(r['final']['step']==10000 for r in runs)
    assert all(np.isfinite(r['final']['loss']) for r in runs)
    config=parts[0]['config']|dict(seeds=3,seed_start=0,output='readout-results.json',
        execution_note='Independent seeds run concurrently with CPU affinities 0-5, 6-11, 12-17. Timings are not a speed comparison.')
    Path('readout-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    lines=['Queried copying, context 16, 10,000 updates, three seeds.',
           'Lossless block encoding width 255; baseline hidden width 240.',
           'Final accuracy: mean +/- sample standard deviation (%).','',
           'Variant                 Accuracy          Trainable parameters']
    for variant in VARIANTS:
        rs=[r for r in runs if r['variant']==variant]
        xs=[100*r['final']['accuracy'] for r in rs]
        lines.append(f"{variant:24s}{np.mean(xs):6.2f} +/- {np.std(xs,ddof=1):5.2f}    {rs[0]['trainable_parameters']:,}")
    lines.extend(['','Successful frozen-readout continuation (3,000 further updates, Adam 0.0001):'])
    for variant in VARIANTS:
        for unfreeze in [False,True]:
            xs=[100*c['final']['accuracy'] for r in runs if r['variant']==variant
                for c in r.get('continuation',[]) if c['unfreeze']==unfreeze]
            if xs:
                lines.append(f'{variant} unfreeze={unfreeze}: {np.mean(xs):.2f} +/- {np.std(xs,ddof=1):.2f} (n={len(xs)})')
    small=Path('readout-small-pilot.json')
    if small.exists():
        data=json.loads(small.read_text())
        lines.extend(['','Smaller-model pilot: width 16, hidden 72, ONE SEED only.'])
        for r in data['runs']:
            lines.append(f"{r['variant']}: {100*r['final']['accuracy']:.2f}%")
    text='\n'.join(lines)+'\n'
    Path('readout-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
