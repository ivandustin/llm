"""Validate the retained-model language test and preserve usable model metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parts=[json.loads(Path(f'language-depth-seed-{s}.json').read_text()) for s in range(3)]
    config=parts[0]['config']
    for p in parts:
        for key in ['steps','context','vocab','batch','lr','dataset_sha256','split_lengths','evaluation_seed','evaluation_examples_per_split','deep','shallow','transformer']:
            assert p['config'][key]==config[key],key
    previous=json.loads(Path('results.json').read_text())
    for key in ['steps','context','vocab','batch','lr','dataset_sha256','split_lengths']:
        assert config[key]==previous['config'][key],key
    runs=[r for p in parts for r in p['runs']]
    assert len(runs)==9 and {(r['model'],r['seed']) for r in runs}=={(m,s) for m in ['deep','shallow','transformer'] for s in range(3)}
    assert all(r['parameters']==config[r['model']]['parameters']<=10000 for r in runs)
    assert all(r['history'][-1]['step']==10000 for r in runs)
    assert all(np.isfinite(r['test']['loss']) and np.isfinite(r['test']['perplexity']) for r in runs)
    config=config|dict(seeds=3,output='language-depth-results.json',execution_note='Independent seeds ran concurrently; timing fields are diagnostic only, not isolated speed comparisons.')
    config.pop('seed',None)
    Path('language-depth-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    rows=[]
    for name in ['deep','shallow','transformer']:
        rs=[r for r in runs if r['model']==name]
        row=dict(model=name,parameters=rs[0]['parameters'])
        for split in ['validation','test']:
            row[split]={}
            for key in ['loss','perplexity','accuracy']:
                vals=[r[split][key] for r in rs]
                row[split][key]=dict(mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),individual=vals)
        rows.append(row)
    summary=dict(config=config,rows=rows,
        interpretation='Architecture selected on queried copying; this is a fresh-weight language test, not language-specific architecture tuning.')
    Path('language-depth-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    raw=(Path(__file__).resolve().parents[1] / 'data/shakespeare.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==config['dataset_sha256']
    vocabulary=dict(characters=sorted(set(raw.decode())),encoding='zero-based index in characters',dataset_sha256=config['dataset_sha256'])
    Path('language-vocabulary.json').write_text(json.dumps(vocabulary,indent=2)+'\n')
    manifest_path=Path('best-model.json')
    manifest=json.loads(manifest_path.read_text())
    manifest['language_test'].update(dict(status='complete',results='language-depth-summary.json',vocabulary='language-vocabulary.json',
        checkpoints=[f'language-depth-seed-{s}/deep-seed-{s}.npz' for s in range(3)],
        selection_note='Original queried-copy selection retained; language metrics are reported separately.'))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['Retained six-layer model: next-character language test, Tiny Shakespeare.',
        'Context 32, 65 characters; contiguous 80/10/10 split.',
        'Adam 0.001, batch 64, 10,000 compiled updates, three seeds.',
        'Same training batches and fixed held-out windows as the earlier language benchmark.',
        'Means +/- sample standard deviations across seeds.',
        '', 'Model          Parameters   Test perplexity    Test accuracy (%)']
    for row in rows:
        pp=row['test']['perplexity'];ac=row['test']['accuracy']
        lines.append(f"{row['model']:14s}{row['parameters']:9,d}    {pp['mean']:.2f} +/- {pp['sd']:.2f}     {100*ac['mean']:.2f} +/- {100*ac['sd']:.2f}")
    lines.extend(['','Model definitions:',
        'deep: embedding width 64; six hidden layers of width 12; normalized GELU; no residuals or gate.',
        'shallow: embedding width 64; one hidden layer of width 18; normalized GELU; exact same parameter count.',
        'transformer: one block, width 16, four heads, feed-forward width 144.',
        '', 'Samples (temperature 0.8, seed 0, 160 generated characters):'])
    for name in ['deep','shallow','transformer']:
        lines.extend(['',name+':',next(r['sample'] for r in runs if r['model']==name and r['seed']==0)])
    text='\n'.join(lines)+'\n'
    Path('language-depth-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
