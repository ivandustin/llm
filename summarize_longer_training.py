"""Compare predeclared10k/30k/50k checkpoints of our original six-layer model."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs=[]
for seed in range(3):
 part=json.loads(Path(f'longer100k-seed-{seed}.json').read_text());configs.append(part['config'])
 assert len(part['runs'])==4
 for r in part['runs']:
  assert r['complete'] and r['updates_completed']==50000
  assert [p['step'] for p in r['checkpoints']]==[10000,30000,50000]
  assert r['checkpoints'][0]['baseline_check']
  assert all(np.isfinite(p['final']['loss']) for p in r['checkpoints'])
  runs.append(r)
for task in ['position','single','multi','language']:
 for step in [10000,30000,50000]:
  rs=[next(p['final'] for p in r['checkpoints'] if p['step']==step) for r in runs if r['task']==task]
  acc=[100*r['accuracy'] for r in rs];ppl=[r['perplexity'] for r in rs]
  rows.append(dict(task=task,step=step,accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('longer100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Longer training, only original pooled model: d64,h120,six hidden layers.', 'GELU,learned token/position embeddings,parameter-free pooled norm,no residual/gate/final norm.', 'Query parameters99184; language94473. Three seeds,Adam .001,batch64,no schedule/weight decay.', 'Train from the original initialization; the first10k updates reproduce prior loss/accuracy for every task/seed.', 'Checkpoints10k/30k/50k are predeclared; no best-step selection. Same validation/test examples as previous experiments.', 'Data generated in1000-update chunks, continuous Adam/model state throughout. Only weight files are saved to disk.', 'The50k pooled runs use five times the updates/examples of the prior10k Transformer runs; this is not a matched-training Transformer comparison. No Transformer runs or speed benchmark.','']
for r in rows:
 lines.append(f"{r['task']} / {r['step']} updates: accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if r['task']=='language' else ''))
t='\n'.join(lines)+'\n';Path('longer100k-summary.txt').write_text(t);print(t)
with Path('README.md').open('a') as f:f.write('\n## Original six-layer pooling: longer training\n\n'+t+'\nReproduce with `longer_training_100k.py --seed 0|1|2`, then `summarize_longer_training.py`. Predeclared50k updates and data RNG streams match the original first10k batches. All model parameters and Adam accumulators carry between scans, with no checkpoint restart/reset. JSON progress is saved each1000-update chunk; NPZ files at10k/30k/50k contain model weights only. No paper/default-model changes made.\n')
