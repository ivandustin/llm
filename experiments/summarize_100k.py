"""Validate the shared-architecture ~100k language/query comparison."""
import json
from pathlib import Path
import numpy as np


def main():
 rows=[];runs=[]
 for arch in ['ours','transformer']:
  parts=[json.loads(Path(f'large-{arch}-seed-{s}.json').read_text()) for s in range(3)]
  rs=[r for p in parts for r in p['runs']]
  assert len(rs)==12 and {(r['task'],r['seed']) for r in rs}=={(t,s) for t in ['position','single','multi','language'] for s in range(3)}
  assert all(r['parameters']<=100000 and r['history'][-1]['step']==10000 and np.isfinite(r['final']['loss']) for r in rs)
  for t in ['position','single','multi','language']:
   a=[r for r in rs if r['task']==t]
   acc=[100*r['final']['accuracy'] for r in a]
   pp=[r['final']['perplexity'] for r in a]
   rows.append(dict(model=arch,task=t,parameters=a[0]['parameters'],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(pp)),perplexity_sd=float(np.std(pp,ddof=1))))
  runs.extend(dict(r,architecture=arch) for r in rs)
 out=dict(config=dict(ours=dict(width=64,hidden=120,hidden_layers=6,residual=False),transformer=dict(width=64,hidden=536,hidden_layers=1,blocks=1,heads=4),steps=10000,batch=64,optimizer='Adam',lr=.001,seeds=3,training='compiled lax.scan'),rows=rows,runs=runs)
 Path('100k-results.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['Shared architecture across tasks, approximately 100k parameters.',
 'Ours: 64 dims, six hidden layers of 120 units, normalization + GELU, no residuals.',
 'Transformer: 64 dims, one block, four heads, standard one-hidden-layer FFN of 536 units.',
 'Three seeds, 10,000 compiled Adam updates, batch 64, learning rate .001.',
 'Same data and evaluation samples as previous tests. No speed benchmark.',
 'Parameter counts change with context/vocabulary; internal shapes stay fixed.','']
 for t in ['position','single','multi','language']:
  for r in [v for v in rows if v['task']==t]:
   lines.append(f"{t} / {r['model']}: params={r['parameters']}, accuracy={r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity={r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if t=='language' else ''))
 text='\n'.join(lines)+'\n';Path('100k-summary.txt').write_text(text);print(text)
 with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Approximately 100k parameters, one-block transformer\n\n'+text+'\nLanguage uses the previous Tiny Shakespeare split and context 32. Position copying\nuses context 16. Single and multiple associative queries use five stored pairs\nand causal prefixes, as in `mqar_test.py`. Single queries supervise one label\nper context; multi queries supervise three. The paper remains paused. This\nretains the one-block standard-FFN transformer requested by the user and does\nnot claim an exhaustive transformer architecture search. Final checkpoints\nare scored, with no best-step selection. Runs and weights are in `large-*`;\n`benchmark_100k.py` and `summarize_100k.py` reproduce the experiment.\n')


if __name__=='__main__':main()
