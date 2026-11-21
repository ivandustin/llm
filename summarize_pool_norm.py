"""Compare pooled normalization and activations at the original diagnostic sizes."""
import json
from pathlib import Path
import numpy as np
variants=['sigmoid','norm_sigmoid','affine_norm_sigmoid','gelu','norm_gelu','affine_norm_gelu']
mapping=dict(sigmoid='random_joint',norm_sigmoid='random_norm',gelu='random_gelu',norm_gelu='random_norm_gelu')
rows=[];runs=[]
for task in ['position','language']:
 for variant in variants:
  selected=[]
  for seed in range(3):
   if task=='position' and not variant.startswith('affine_'):
    part=json.loads(Path(f'readout-seed-{seed}.json').read_text())
    r=next(r for r in part['runs'] if r['variant']==mapping[variant])
    metric=r['final'];parameters=r['trainable_parameters'];source=f'readout-seed-{seed}.json';mode='historical final 4096-example diagnostic evaluation'
   else:
    part=json.loads(Path(f'poolnorm-{variant}-seed-{seed}.json').read_text())
    r=next(r for r in part['runs'] if r['task']==task)
    assert r['history'][-1]['step']==10000
    metric=r['history'][-1] if task=='position' else r['final']
    parameters=r['parameters'];source=f'poolnorm-{variant}-seed-{seed}.json';mode='matched 4096-example diagnostic evaluation' if task=='position' else '4096-example language test'
   assert np.isfinite(metric['loss'])
   selected.append(metric);runs.append(dict(task=task,variant=variant,seed=seed,parameters=parameters,metrics=metric,source=source,evaluation=mode))
  acc=[100*r['accuracy'] for r in selected];ppl=[np.exp(r['loss']) for r in selected]
  rows.append(dict(task=task,variant=variant,parameters=parameters,accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
config=dict(position=dict(width=255,hidden=240,context=16,vocab=96),language=dict(width=32,hidden=152,context=32,vocab=65),hidden_layers=1,norm_location='pooled vector, before input linear',norm_epsilon=1e-5,affine='learned per-coordinate scale and bias, initialized1/0',learned_positions=True,final_norm=False,steps=10000,seeds=[0,1,2],batch=64,optimizer='Adam',lr=.001,training='new runs compiled lax.scan; historical copy controls ordinary JIT updates',copy_controls='reused historical random-embedding results, same 4096 examples',language_controls='all six rerun',added_parameters=dict(position=510,language=64))
Path('poolnorm-results.json').write_text(json.dumps(dict(config=config,rows=rows,runs=runs),indent=2)+'\n')
lines=['Pooled learned-affine LayerNorm comparison, one hidden layer, learned positions.', 'Query-copy: d255 h240, 113136 base params / 113646 affine params.', 'Language: d32 h152, 18065 base params / 18129 affine params.', 'Three seeds, 10000 Adam updates, batch64, lr .001. Epsilon1e-5; final norm unchanged (none).', 'Query controls reuse original four random-embedding diagnostics. New copy scores use the identical 4096 evaluation examples; extra fresh16384 scores are in raw files.', 'All six language variants rerun with identical data, held-out windows and shared initialization.', 'No speed benchmark.','']
for task in ['position','language']:
 for r in [x for x in rows if x['task']==task]:
  lines.append(f"{task} / {r['variant']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('poolnorm-summary.txt').write_text(t);print(t)
with Path('README.md').open('a') as f:f.write('\n## Learned affine pooled LayerNorm, query-copy and language\n\n'+t+'\nImplementation: `learned_pool_norm_test.py`; aggregation: `summarize_pool_norm.py`. `check_learned_pool_norm.py` verifies identical shared parameters and initial logits between parameter-free and affine normalization, expected extra parameter counts, and five-step full compiled versus ordinary loss/Adam-state parity for sigmoid and GELU. Learned scale/bias use the exact previous normalization expression, then per-coordinate affine transformation, avoiding a changed normalization kernel/epsilon as a confound. Main architecture and paper remain unchanged.\n')
