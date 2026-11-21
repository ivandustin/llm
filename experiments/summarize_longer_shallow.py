"""Longer single-hidden-layer training and six-layer controls at same updates."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs={}
for arch,prefix in [('shallow','longer-shallow100k'),('deep','longer100k')]:
 for seed in range(3):
  part=json.loads(Path(f'{prefix}-seed-{seed}.json').read_text());configs[arch]=part['config'];assert len(part['runs'])==4
  for r in part['runs']:
   assert r['complete'] and r['updates_completed']==50000
   assert [p['step'] for p in r['checkpoints']]==[10000,30000,50000] and r['checkpoints'][0]['baseline_check']
   assert all(np.isfinite(p['final']['loss']) for p in r['checkpoints'])
   runs.append(dict(r,architecture=arch))
 for task in ['position','single','multi','language']:
  for step in [10000,30000,50000]:
   rs=[next(p['final'] for p in r['checkpoints'] if p['step']==step) for r in runs if r['architecture']==arch and r['task']==task]
   acc=[100*r['accuracy'] for r in rs];ppl=[r['perplexity'] for r in rs]
   rows.append(dict(architecture=arch,task=task,step=step,accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('longer-shallow100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Longer training: one hidden layer d64 h576 versus six layers d64 h120.', 'Both: learned positions,GELU,parameter-free pooled norm,no residual/gate/final norm.', 'Shallow parameters100000 queries/81153 language; deep99184 queries/94473 language.', 'Three seeds,Adam .001,batch64,no schedule/weight decay. Continuous model/Adam state in1000-update scans.', 'First10k reproduces previous model-specific loss/accuracy in every task/seed. Same generated streams,held-out samples and language splits.', 'Checkpoints10k/30k/50k predeclared; no best-checkpoint selection. Deep50k results reused; only shallow newly trained.', 'Different hidden shape and initialization layout; language parameter counts not exactly matched. No Transformer runs or speed measurements.','']
for task in ['position','single','multi','language']:
 for arch in ['shallow','deep']:
  for r in [v for v in rows if v['task']==task and v['architecture']==arch]:
   lines.append(f"{task} / {arch} / {r['step']} updates: accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('longer-shallow100k-summary.txt').write_text(t);print(t)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Longer training of the single-hidden-layer control\n\n'+t+'\nReproduce via `longer_training_100k.py --architecture shallow --seed 0|1|2`, then `summarize_longer_shallow.py`. The shallow implementation is the earlier n1 control in `parallel_mixture_100k.py`, with one branch and one hidden layer (h576). Original deep long-run commands remain supported with the default architecture deep. All12 new10k model/task/seed baseline checks must pass. Saved NPZ checkpoints contain weights only, not optimizer state; Adam is uninterrupted in memory. No paper/default-model changes made.\n')
