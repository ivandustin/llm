"""Aggregate fixed-position comparison without changing the paper."""
import json
from pathlib import Path
import numpy as np
rows=[];runs=[]
for arch in ['ours','transformer']:
 for seed in range(3):
  part=json.loads(Path(f'sinusoidal-{arch}-seed-{seed}.json').read_text())
  assert len(part['runs'])==4
  for run in part['runs']:
   assert run['parameters']<=10000 and run['history'][-1]['step']==10000
   assert np.isfinite(run['final']['loss'])
   runs.append(dict(run,architecture=arch))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['architecture']==arch and r['task']==task]
  assert len(rs)==3
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(model=arch,task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
config=dict(positions='fixed interleaved sin/cos, base10000, zero trainable position parameters',budget=10000,ours=dict(width=64,layers=6,hidden_query=15,hidden_language=23,residual=False),transformer=dict(width=16,blocks=1,heads=4,ff_hidden_query=170,ff_hidden_language=201),steps=10000,batch=64,optimizer='Adam',lr=.001,seeds=[0,1,2],training='compiled lax.scan',context_language=32,context_queries=16)
Path('sinusoidal-results.json').write_text(json.dumps(dict(config=config,rows=rows,runs=runs),indent=2)+'\n')
lines=['Fixed sinusoidal positions, <=10k trainable parameters, three seeds.', 'Mean +/- sample standard deviation; 10,000 Adam updates, batch64, lr .001.', 'Ours: d64, six hidden layers, h15 queries / h23 language; no residuals.', 'Transformer: d16, four heads, one standard block, FF170 queries / FF201 language.', 'Same data/evaluation as previous tests. Hidden widths use the freed positional budget.', 'This is a matched-budget comparison, not an isolated positional-encoding ablation.', 'No speed measurement or long-context evaluation.','']
for task in ['position','single','multi','language']:
 for r in [x for x in rows if x['task']==task]:
  lines.append(f"{task} / {r['model']}: {r['parameters']} params, accuracy {r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity {r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
text='\n'.join(lines)+'\n';Path('sinusoidal-summary.txt').write_text(text);print(text)
with Path('README.md').open('a') as f:f.write('\n## Fixed sinusoidal positions, approximately 10k parameters\n\n'+text+'\nReproduce with `sinusoidal_test.py --architecture ours|transformer --seed 0|1|2`, then `summarize_sinusoidal.py`. Logit/gradient parity for the optimized Transformer and complete five-step compiled/ordinary optimizer-state parity for both models pass in `check_sinusoidal.py`. Position weights multiply token embeddings in our model and add to token embeddings in the Transformer. Learned embeddings are replaced, not retained. All trainable weights are counted; fixed sinusoids contribute zero parameters. No paper changes were made.\n')
