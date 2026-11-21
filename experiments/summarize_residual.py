"""Summarize paired residual accuracy and previously measured isolated timings."""
import json
from pathlib import Path
import numpy as np
from experiments.benchmark import nnx
from experiments.capabilities import ProbeTransformer,generate
from experiments.readout_experiments import metrics
from experiments.validate_layer_depth import load_weights


def main():
    parts=[json.loads(Path(f'residual-seed-{s}.json').read_text()) for s in range(3)]
    config=parts[0]['config']
    for p in parts:
        for key in ['steps','context','vocab','width','hidden','hidden_layers','parameters','budget','batch','lr','evaluation_seed','validation_seed','validation_examples']:
            assert p['config'][key]==config[key],key
    runs=[r for p in parts for r in p['runs']]
    assert {(r['variant'],r['seed']) for r in runs}=={(v,s) for v in ['plain','residual'] for s in range(3)}
    assert len(runs)==6 and all(r['parameters']==9976 and r['final']['step']==10000 for r in runs)
    assert all(np.isfinite(r['validation']['loss']) for r in runs)
    config=config|dict(seeds=3,output='residual-results.json')
    config.pop('seed',None)
    Path('residual-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    reference_path=Path('residual-transformer-validation.json')
    if reference_path.exists():
        saved=json.loads(reference_path.read_text())
        assert saved['evaluation_seed']==config['validation_seed'] and saved['examples']==config['validation_examples']
        refs=saved['runs']
        assert len(refs)==3 and {r['seed'] for r in refs}=={0,1,2}
    else:
        x,y,_=generate('indexed_copy',16,config['validation_examples'],np.random.default_rng(config['validation_seed']))
        refs=[]
        for seed in range(3):
            model=ProbeTransformer(96,16,16,nnx.Rngs(seed),depth=1,ff_hidden=162)
            load_weights(model,Path(f'budget-seed-{seed}')/f'transformer-width-16-seed-{seed}.npz')
            refs.append(dict(seed=seed,parameters=9970,validation=metrics(model,x,y)))
        reference_path.write_text(json.dumps(dict(evaluation_seed=config['validation_seed'],examples=config['validation_examples'],runs=refs),indent=2)+'\n')
    rows=[]
    for variant in ['plain','residual']:
        rs=[r for r in runs if r['variant']==variant]
        row=dict(model=variant,parameters=9976)
        for key in ['final','validation']:
            a=[100*r[key]['accuracy'] for r in rs]
            row[key]=dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),individual=a)
        rows.append(row)
    a=[100*r['validation']['accuracy'] for r in refs]
    rows.append(dict(model='transformer',parameters=9970,validation=dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),individual=a)))
    speed=json.loads(Path('residual-speed-results.json').read_text())
    assert speed['config']['include_residual'] and speed['config']['trials']==7
    assert speed['config']['parameters']==dict(candidate=9976,residual=9976,transformer=9970)
    summary=dict(config=config,accuracy=rows,speed=speed['results'],speed_config=speed['config'])
    Path('residual-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Residual ablation: six hidden layers of width 12, embedding width 64.',
        'Adam 0.001, batch 64, 10,000 updates, three seeds, 9,976 parameters.',
        'Paired models have identical initial weights and training batches.',
        'Both trained in compiled lax.scan; five-step ordinary/compiled parity checks passed.',
        '', 'Model        Original evaluation accuracy   New 16,384-example accuracy']
    for r in rows:
        old=r.get('final')
        old_text=f"{old['mean']:.2f} +/- {old['sd']:.2f}" if old else 'saved reference: 100%'
        v=r['validation']
        lines.append(f"{r['model']:12s} {old_text:29s} {v['mean']:.2f} +/- {v['sd']:.2f}")
    lines.extend(['','Previously completed isolated timing: median ms, seven trials, full CPU affinity.',
                  'Mode                              Plain     Residual   Transformer'])
    for mode,values in speed['results'].items():
        for v in values.values():
            assert len(v['seconds_per_step_trials'])==7 and all(np.isfinite(x) and x>0 for x in v['seconds_per_step_trials'])
        lines.append(f"{mode:33s} {values['candidate']['milliseconds_per_step_median']:8.3f} {values['residual']['milliseconds_per_step_median']:10.3f} {values['transformer']['milliseconds_per_step_median']:11.3f}")
    text='\n'.join(lines)+'\n'
    Path('residual-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
