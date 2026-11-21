"""Aggregate six-layer wider pooling with reused d64 control."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[];configs={}
for arch in ['w64','w128','w256','w512']:
 for seed in range(3):
  source=f'large-ours-seed-{seed}.json' if arch=='w64' else f'width100k-{arch}-seed-{seed}.json'
  part=json.loads(Path(source).read_text());configs[arch]=part['config'];assert len(part['runs'])==4
  for r in part['runs']:
   assert r['history'][-1]['step']==10000 and r['parameters']<=100000 and np.isfinite(r['final']['loss'])
   runs.append(dict(r,architecture=arch,source=source))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['architecture']==arch and r['task']==task]
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(architecture=arch,width=int(arch[1:]),hidden=configs[arch]['hidden'],task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('width100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Six-layer pooling, wider embedding dimensions at <=100k total parameters.', 'Learned token/position embeddings, GELU, non-learned pooled norm epsilon1e-5, no residual/gate/final norm.', 'Embedding/hidden widths:64/120,128/109,256/88,512/49. Same shapes across query/language tasks.', 'Widths chosen to maximize hidden width under the query budget; language counts differ because vocabulary/context differ.', 'Three seeds,10000 compiled Adam updates,batch64,lr .001,same training/evaluation samples. Final checkpoint only.', 'Prior d64 controls reused. Context32 language; context16 retrieval. No speed test or new Transformer runs.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / d{r['width']} h{r['hidden']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('width100k-summary.txt').write_text(t);print(t)
with Path('README.md').open('a') as f:f.write('\n## Six-layer wider embeddings under100k parameters\n\n'+t+'\nReproduce using `width_100k_test.py --architecture w128|w256|w512 --seed 0|1|2`, then `summarize_width_100k.py`. `check_width_100k.py` verifies total parameter formula/count and five-step complete compiled losses/model/Adam-state parity at each width. This reallocates a fixed total budget between embeddings and hidden layers; it is not an increase of d at fixed hidden width. No automatic paper/default-architecture changes made.\n')
