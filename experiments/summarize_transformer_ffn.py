"""Compare the six-hidden-layer transformer with saved same-task controls."""
import json
from pathlib import Path
import numpy as np


def avg(values):return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))

def main():
 parts=[json.loads(Path(f'transformer-deep-ffn-seed-{s}.json').read_text()) for s in range(3)]
 runs=[r for p in parts for r in p['runs']]
 assert len(runs)==12 and {(r['task'],r['seed']) for r in runs}=={(t,s) for t in ['position','single','multi','language'] for s in range(3)}
 assert all(r['history'][-1]['step']==10000 and np.isfinite(r['final']['loss']) for r in runs)
 position=json.loads(Path('residual-summary.json').read_text())['accuracy']
 assoc=json.loads(Path('associative-results.json').read_text())['rows']
 language=json.loads(Path('language-depth-summary.json').read_text())['rows']
 rows=[]
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['task']==task]
  assert all(r['parameters']==(8841 if task=='language' else 9608) for r in rs)
  if task=='position':
   ours=next(r for r in position if r['model']=='plain')['validation'];old=next(r for r in position if r['model']=='transformer')['validation']
  elif task in ['single','multi']:
   a=next(r for r in assoc if r['task']==task and r['model']=='ours');b=next(r for r in assoc if r['task']==task and r['model']=='transformer')
   ours=dict(mean=a['accuracy_mean'],sd=a['accuracy_sd']);old=dict(mean=b['accuracy_mean'],sd=b['accuracy_sd'])
  else:
   ours=next(r for r in language if r['model']=='deep')['test']['perplexity'];old=next(r for r in language if r['model']=='transformer')['test']['perplexity']
  metric='perplexity' if task=='language' else 'accuracy'
  new=avg([r['final'][metric]*(1 if task=='language' else 100) for r in rs])
  row=dict(task=task,metric=metric,ours=ours,transformer_one_hidden=old,transformer_six_hidden=new,parameters=rs[0]['parameters'])
  if task=='language':row['new_language_accuracy']=avg([100*r['final']['accuracy'] for r in rs])
  rows.append(row)
 out=dict(config=parts[0]['config'],rows=rows,runs=runs,
  note='Identical depth/width within each architecture across tasks; vocabulary/context change parameters. Transformer differs by <4% from ours; same data, batches, updates and seeds.')
 Path('transformer-ffn-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['Six feed-forward hidden layers in both models; one attention block in transformer.',
  'Ours: embedding 64 / hidden 12. Transformer: embedding 16 / hidden 28.',
  'Queries: ours 9,976 params; transformer 9,608. Language: ours 8,613; transformer 8,841.',
  'Three seeds, 10,000 Adam updates, batch 64, identical data per task.', '',
  'Task/metric         Ours              Transformer 1 hidden  Transformer 6 hidden']
 for r in rows:
  f=lambda v:f"{v['mean']:.2f} +/- {v['sd']:.2f}"
  lines.append(f"{r['task']+'/'+r['metric']:20s}{f(r['ours']):18s}{f(r['transformer_one_hidden']):22s}{f(r['transformer_six_hidden'])}")
 text='\n'.join(lines)+'\n';Path('transformer-ffn-comparison.txt').write_text(text);print(text)
 with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## Six-hidden-layer transformer FFN comparison\n\n'+text+'\nThe transformer retains one causal attention block, prenorm and block residuals.\nIts FFN has an input projection, five equal-width hidden projections, and an\noutput projection, with GELU after each hidden layer. All seven affine maps\nare counted. No intermediate FFN normalizations or skips are added. Both\narchitectures use the same internal shapes across tasks. Total parameters are\nclose rather than exact (difference below 4%). Earlier one-hidden-layer\ntransformer and our-model controls are reused. This is not an official MQAR\nbenchmark, a multi-attention-block test, or a speed comparison.\n')


if __name__=='__main__':main()
