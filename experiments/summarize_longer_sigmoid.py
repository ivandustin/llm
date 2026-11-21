"""Original sigmoid/no-norm versus paired GELU/free-norm shallow control."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs={}
for arch,prefix in [('sigmoid_no_norm','longer-sigmoid100k'),('gelu_norm','longer-shallow100k')]:
 for seed in range(3):
  part=json.loads(Path(f'{prefix}-seed-{seed}.json').read_text());configs[arch]=part['config'];assert len(part['runs'])==4
  for r in part['runs']:
   assert r['complete'] and r['updates_completed']==50000
   assert [p['step'] for p in r['checkpoints']]==[10000,30000,50000]
   assert all(np.isfinite(p['final']['loss']) for p in r['checkpoints'])
   runs.append(dict(r,architecture=arch))
 for task in ['position','single','multi','language']:
  for step in [10000,30000,50000]:
   rs=[next(p['final'] for p in r['checkpoints'] if p['step']==step) for r in runs if r['architecture']==arch and r['task']==task]
   acc=[100*r['accuracy'] for r in rs];ppl=[r['perplexity'] for r in rs]
   rows.append(dict(architecture=arch,task=task,step=step,accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('longer-sigmoid100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Original sigmoid/no-normalization versus GELU/parameter-free pooled normalization.', 'One hidden layer,d64,h576,learned positions,no residual/gate/final norm.', 'Same initial parameters,data streams,held-out examples and parameter counts:100000 queries/81153 language.', 'Three seeds,Adam .001,batch64,1000-update compiled chunks,continuous model/Adam state,no schedule/weight decay.', 'Predeclared checkpoints10k/30k/50k; no best-step selection. Only sigmoid/no-norm newly trained; GELU/norm controls reused.', 'Activation and pooled norm both change; this does not isolate either effect separately. No speed benchmark or Transformer runs.','']
for task in ['position','single','multi','language']:
 for arch in ['sigmoid_no_norm','gelu_norm']:
  for r in [v for v in rows if v['task']==task and v['architecture']==arch]:
   lines.append(f"{task} / {arch} / {r['step']} updates: accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('longer-sigmoid100k-summary.txt').write_text(t);print(t)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Original sigmoid without normalization, longer training\n\n'+t+'\nReproduce with `longer_training_100k.py --architecture sigmoid --seed 0|1|2`, then `summarize_longer_sigmoid.py`. `sigmoid_no_norm.py` implements raw position-token sum pooling, Linear, sigmoid, and output Linear with the exact same parameter layout/initial values as the earlier n1 shallow control. `check_sigmoid_no_norm.py` verifies identical weights/counts, expected unnormalized sigmoid formula, and five-step compiled versus ordinary full losses/model/Adam state for query and language shapes. Saved checkpoints contain model weights only; optimizer state remains continuous in memory. No paper/default changes made.\n')
