"""Aggregate deeper pooling vs two-block Transformer at <=100k parameters."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs={}
for arch in ['six','eight','twelve','transformer']:
 for seed in range(3):
  source=f'large-ours-seed-{seed}.json' if arch=='six' else f'depth100k-{arch}-seed-{seed}.json'
  part=json.loads(Path(source).read_text());configs[arch]=part['config']
  assert len(part['runs'])==4
  for r in part['runs']:
   assert r['history'][-1]['step']==10000 and r['parameters']<=100000 and np.isfinite(r['final']['loss'])
   runs.append(dict(r,architecture=arch,source=source))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['architecture']==arch and r['task']==task]
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(architecture=arch,task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('depth100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Deeper pooled models, <=100k parameters; d64, learned positions.', 'Ours: GELU, parameter-free pooled normalization, no residuals/gates/final norm.', 'Pooled hidden layers/width: 6/120, 8/103, 12/84. Internal dimensions fixed across tasks.', 'Transformer: 2 standard blocks, d64, 4 heads, FF200 per block across all tasks.', 'Three seeds, 10000 compiled Adam updates, batch64, lr .001. Same training/evaluation examples.', 'Six-layer results reused from original100k runs; deeper/2-block reference newly trained. Final checkpoints only.', 'Context32 language; context16 retrieval. No speed measurement.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / {r['architecture']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('depth100k-summary.txt').write_text(t);print(t)
with Path('README.md').open('a') as f:f.write('\n## Deeper pooling at approximately100k parameters\n\n'+t+'\nReproduce with `depth_100k_test.py --architecture eight|twelve|transformer --seed 0|1|2`, then `summarize_depth_100k.py`. `check_depth_100k.py` verifies counts and five-step complete compiled model/Adam state and losses against ordinary updates for all new architectures. Two-block Transformer uses the previously validated full-first-block/selected-last-block implementation, now FF200 for both language and queries. This tests depth allocation at a fixed budget; it does not hold hidden width constant. No automatic paper or default-architecture changes made.\n')
