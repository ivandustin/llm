"""Equal probability mixtures vs the existing six-layer pooled model."""
import json
from pathlib import Path
import numpy as np
rows=[];runs=[];configs={}
for arch in ['deep6','n1','n2','n4','n8']:
 for seed in range(3):
  source=f'large-ours-seed-{seed}.json' if arch=='deep6' else f'deepmixture100k-{arch}-seed-{seed}.json'
  part=json.loads(Path(source).read_text());configs[arch]=part['config'];assert len(part['runs'])==4
  for r in part['runs']:
   assert r['history'][-1]['step']==10000 and r['parameters']<=100000 and np.isfinite(r['final']['loss'])
   runs.append(dict(r,architecture=arch,source=source))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['architecture']==arch and r['task']==task]
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(architecture=arch,task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('deepmixture100k-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Shared d64 position-token pool followed by parallel six-hidden-layer FFNs.', 'Each branch: Linear64->h+GELU, five Linearh->h+GELU, Linearh->V, softmax. Final distribution is equal1/N average of probabilities.', 'Shared learned token/position tables, non-learned pooled norm epsilon1e-5; no residual or final norm.', 'Branch count/hidden width:1/120,2/81,4/53,8/34. All trainable branch weights, biases and shared embeddings counted.', 'Same internal shapes across tasks. Query budget<=100k; language counts lower due to smaller vocabulary/context.', 'Three seeds,10000 compiled Adam updates,batch64,lr .001,same data/evaluation as prior runs. Final checkpoints only.', 'Deep6 reference reused, d64 h120, no residual. Fresh one-branch six-layer control also rerun; parameter initialization layout differs from historical deep6; not identical initial weights across branch counts.', 'Context32 language;16 retrieval. No speed measurement.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / {r['architecture']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
t='\n'.join(lines)+'\n';Path('deepmixture100k-summary.txt').write_text(t);print(t)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Parallel six-layer FFNs with equal averaging of probabilities\n\n'+t+'\nReproduce via `deep_parallel_mixture_100k.py --architecture n1|n2|n4|n8 --seed 0|1|2`, then `summarize_deep_parallel_mixture.py`. `check_deep_parallel_mixture.py` checks parameter counts, arithmetic-mean probabilities and unit total mass, finite gradients for every branch, and five-step complete compiled-vs-ordinary model/Adam losses/state parity. Stable implementation uses logsumexp over branch log-softmax values minus logN, returning normalized log probabilities to the common CE interface. This is a probability mixture, not the mathematically equivalent wide shallow logit-sum architecture initially discussed. No automatic paper or default-architecture changes made.\n')
