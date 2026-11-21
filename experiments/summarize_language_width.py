"""Summarize 16/32-dimension language checks against saved matched controls."""
import json
from pathlib import Path
import numpy as np


def main():
    old=json.loads(Path('language-depth-results.json').read_text())
    old_config=old['config']
    runs=[]
    configs=[]
    for width,hidden,params in [(16,30,8727),(32,24,8521)]:
        for seed in range(3):
            part=json.loads(Path(f'language-width-{width}-seed-{seed}.json').read_text())
            cfg=part['config']
            for key in ['steps','context','vocab','batch','lr','dataset_sha256','split_lengths','evaluation_seed','evaluation_examples_per_split']:
                assert cfg[key]==old_config[key],key
            assert cfg['deep']['width']==width and cfg['deep']['hidden']==hidden
            assert len(part['runs'])==1
            r=part['runs'][0]
            assert r['seed']==seed and r['model']=='deep' and r['parameters']==params<10000
            assert r['history'][-1]['step']==10000 and np.isfinite(r['test']['loss'])
            runs.append(dict(r,width=width,hidden=hidden,hidden_layers=6))
            configs.append(cfg)
    controls=[]
    for r in old['runs']:
        spec=old_config[r['model']]
        controls.append(dict(r,width=spec['width'],hidden=spec.get('hidden',spec.get('ff_hidden')),
            hidden_layers=6 if r['model']=='deep' else 1))
    rows=[]
    groups=[('deep',16),('deep',32),('deep',64),('shallow',64),('transformer',16)]
    combined=runs+controls
    for name,width in groups:
        rs=[r for r in combined if r['model']==name and r['width']==width]
        assert len(rs)==3
        row=dict(model=name,width=width,hidden=rs[0]['hidden'],hidden_layers=rs[0]['hidden_layers'],parameters=rs[0]['parameters'])
        for split in ['validation','test']:
            row[split]={}
            for metric in ['perplexity','accuracy']:
                a=[r[split][metric] for r in rs]
                row[split][metric]=dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)))
        rows.append(row)
    config={k:old_config[k] for k in ['steps','context','vocab','batch','lr','dataset_sha256','split_lengths','evaluation_seed']}
    config.update(dict(new_widths=[16,32],hidden_layers=6,seeds=3,
        allocation='Nearest equal hidden widths to the original 8,613 parameters, with total below 10,000.',
        comparison_note='New widths use 8,727 and 8,521 parameters; controls are reused, no speed measurements.'))
    output=dict(config=config,runs=combined,rows=rows)
    Path('language-width-results.json').write_text(json.dumps(output,indent=2)+'\n')
    Path('language-width-summary.json').write_text(json.dumps(dict(config=config,rows=rows),indent=2)+'\n')
    lines=['Language width check: six hidden layers, normalized GELU, no residuals or gates.',
        'Same Tiny Shakespeare split, context 32, Adam 0.001, batch 64, 10,000 compiled updates.',
        'Three seeds; means +/- sample standard deviations. Controls reused from previous language test.',
        'Hidden widths adjusted to keep approximately 8.6k parameters, all below 10k.',
        '', 'Model        Dims Hidden Layers Parameters  Test perplexity  Test accuracy (%)']
    for r in rows:
        pp=r['test']['perplexity'];a=r['test']['accuracy']
        lines.append(f"{r['model']:12s}{r['width']:5d}{r['hidden']:7d}{r['hidden_layers']:7d}{r['parameters']:11,d}   {pp['mean']:.2f} +/- {pp['sd']:.2f}   {100*a['mean']:.2f} +/- {100*a['sd']:.2f}")
    text='\n'.join(lines)+'\n'
    Path('language-width-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
