"""Matched-update comparison of default2-block Transformer and pooled variants."""
import json
from pathlib import Path
import numpy as np
SOURCES=dict(transformer='longer-transformer100k',deep='longer100k',shallow='longer-shallow100k',sigmoid='longer-sigmoid100k')
runs=[];rows=[];configs={}
for arch,prefix in SOURCES.items():
 for seed in range(3):
  part=json.loads(Path(f'{prefix}-seed-{seed}.json').read_text());configs[arch]=part['config'];assert len(part['runs'])==4
  for r in part['runs']:
   assert r['complete'] and r['updates_completed']==50000
   assert [p['step'] for p in r['checkpoints']]==[10000,30000,50000]
   if arch!='sigmoid':assert r['checkpoints'][0]['baseline_check']
   assert all(np.isfinite(p['final']['loss']) for p in r['checkpoints'])
   runs.append(dict(r,architecture=arch))
 for task in ['position','single','multi','language']:
  for step in [10000,30000,50000]:
   rs=[next(p['final'] for p in r['checkpoints'] if p['step']==step) for r in runs if r['architecture']==arch and r['task']==task]
   acc=[100*r['accuracy'] for r in rs];ppl=[r['perplexity'] for r in rs]
   rows.append(dict(architecture=arch,task=task,step=step,parameters=configs[arch]['parameters']['language' if task=='language' else 'query'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('longer-transformer100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Matched10k/30k/50k update comparison, same data streams and held-out samples within each task/seed.', 'Transformer:2 standard blocks,d64,4 heads,FF200 per block,learned positions,standard learned LayerNorm/residuals;99056 query/96081 language params.', 'Deep pooling:d64,h120,6 hidden layers,GELU+non-learned pooled norm;99184/94473 params.', 'Shallow pooling:d64,h576,1 hidden layer,GELU+non-learned pooled norm;100000/81153 params.', 'Sigmoid pooling:same shallow shape and counts,no normalization,sigmoid.', 'Three seeds,Adam .001,batch64,no schedule/weight decay,continuous model/Adam state in1000-update scans.', 'Only Transformer newly trained; pooled results reused. All predeclared milestones reported; no best-step selection.', 'Parameter budgets approximate; shallow language variants have fewer params. Same updates/examples are not equal compute. No new speed benchmark.','']
for task in ['position','single','multi','language']:
 for arch in SOURCES:
  for r in [v for v in rows if v['task']==task and v['architecture']==arch]:
   lines.append(f"{task} / {arch} / {r['step']} updates: accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('longer-transformer100k-summary.txt').write_text(t);print(t)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Default two-block Transformer, matched longer training\n\n'+t+'\nReproduce with `longer_training_100k.py --architecture transformer --seed 0|1|2`, then `summarize_longer_transformer.py`. Uses the previously validated two-block implementation from `depth_100k_test.py`, computing all causal states in block1 and only supervised outputs in block2. The10k checkpoint reproduces its historical model/task/seed loss and accuracy before continuing. Models are freshly initialized, and optimizer state never resets within a run. Saved NPZ milestones contain model weights only.\n')
