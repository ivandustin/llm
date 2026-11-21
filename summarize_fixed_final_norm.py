"""Summarize the paired final-normalization ablation."""
import json
from pathlib import Path
import numpy as np
runs=[];rows=[]
for variant in ['plain','final','fixed']:
 for seed in range(3):
  part=json.loads(Path(f'finalnorm-{variant}-seed-{seed}.json').read_text())
  assert len(part['runs'])==4
  for r in part['runs']:
   assert r['history'][-1]['step']==10000 and r['parameters']<=10000
   assert np.isfinite(r['final']['loss'])
   runs.append(dict(r,variant=variant))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['variant']==variant and r['task']==task]
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(variant=variant,task=task,parameters=rs[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
config=dict(width=64,hidden=12,hidden_layers=6,positions='learned',pooled_norm='parameter-free, unchanged',residual=False,gate=False,final_norm='none, learned affine, or parameter-free; epsilon1e-6; after last GELU before output linear',added_parameters=dict(plain=0,final=24,fixed=0),seeds=[0,1,2],steps=10000,batch=64,optimizer='Adam',lr=.001,training='compiled lax.scan',paired='identical shared initialization and data; plain baseline rerun',language_context=32,query_context=16)
Path('finalnorm-fixed-results.json').write_text(json.dumps(dict(config=config,rows=rows,runs=runs),indent=2)+'\n')
lines=['Paired final LayerNorm ablation: learned positions, d64, h12, six hidden layers.', 'Three seeds, 10,000 compiled Adam updates, batch64, lr .001.', 'Variants: plain=no final norm; final=learned scale/bias (+24 params); fixed=no scale/bias (+0 params). All final norms use epsilon1e-6.', 'Mean +/- sample standard deviation. No speed benchmark.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / {r['variant']}: {r['parameters']} params, accuracy {r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity {r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
text='\n'.join(lines)+'\n';Path('finalnorm-fixed-summary.txt').write_text(text);print(text)
with Path('README.md').open('a') as f:f.write('\n## Parameter-free final LayerNorm comparison\n\n'+text+'\nReproduce using `final_norm_test.py --architecture plain|final|fixed --seed 0|1|2` and `summarize_fixed_final_norm.py`. Previous paired plain/learned results are reused; the parameter-free variant uses identical shared initial weights, batches, and evaluation examples. `check_final_norm.py` verifies baseline output parity, identical shared initial parameters, +24 parameters, and five-step compiled versus ordinary loss and complete optimizer/model-state parity for all three variants. Data/evaluation protocol matches `benchmark_100k.py`; architecture is the retained small model. No paper changes were made.\n')
