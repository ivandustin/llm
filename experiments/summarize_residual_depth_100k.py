"""Residual-only runs compared with prior identical-shape plain models."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs={}
for arch in ['six','eight','twelve']:
 for residual in [False,True]:
  variant=f'{arch}-'+('residual' if residual else 'plain')
  for seed in range(3):
   source=f'resdepth100k-{arch}-seed-{seed}.json' if residual else f'large-ours-seed-{seed}.json' if arch=='six' else f'depth100k-{arch}-seed-{seed}.json'
   part=json.loads(Path(source).read_text());configs[variant]=part['config'];assert len(part['runs'])==4
   for r in part['runs']:
    assert r['history'][-1]['step']==10000 and r['parameters']<=100000 and np.isfinite(r['final']['loss'])
    runs.append(dict(r,architecture=arch,residual=residual,variant=variant,source=source))
  for task in ['position','single','multi','language']:
   rs=[r for r in runs if r['variant']==variant and r['task']==task]
   acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
   rows.append(dict(variant=variant,task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('resdepth100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Residual ablation in our pooled model only: approximately100k total parameters.', 'd64, learned token/position embeddings, GELU, non-learned pooled norm epsilon1e-5, no gate/final norm.', 'Depth/hidden width:6/120,8/103,12/84, identical across tasks.', 'Residual rule: h <- h + GELU(W h + b), at the K-1 equal-width hidden transitions; no residual on input/output projections, no added norm or scaling.', 'Residuals add zero parameters. Each residual configuration starts with exactly the same learned weights as its prior plain control.', 'Three seeds,10000 compiled Adam updates,batch64,lr .001,same training/evaluation data. Final checkpoint only.', 'Historical plain controls reused. Context32 language; context16 retrieval. No Transformer runs or speed measurements.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / {r['variant']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('resdepth100k-summary.txt').write_text(t);print(t)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Residuals in6/8/12-layer pooled models at approximately100k\n\n'+t+'\nReproduce via `residual_depth_100k_test.py --architecture six|eight|twelve --seed 0|1|2`, then `summarize_residual_depth_100k.py`. `check_residual_depth_100k.py` verifies identical shared initial parameters, unchanged counts, and five-step compiled versus ordinary losses and full model/Adam state for every depth. Internal configuration fixed across language/query tasks; all trainable parameters counted. No automatic default-model or paper change.\n')
