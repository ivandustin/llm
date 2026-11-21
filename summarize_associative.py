"""Compare single and multiple causal associative queries with shortcuts."""
import json
from pathlib import Path
import numpy as np
from mqar_test import generate


def main():
 rows=[];all_runs=[]
 x,y=generate(16384,np.random.default_rng(719002))
 mode=np.stack([np.bincount(row,minlength=96) for row in x[:,1:10:2]]).argmax(-1)
 for task,prefix,q in [('single','associative-single',1),('multi','mqar',3)]:
  parts=[json.loads(Path(f'{prefix}-seed-{s}.json').read_text()) for s in range(3)]
  rs=[r for p in parts for r in p['runs']]
  assert len(rs)==6 and {(r['model'],r['seed']) for r in rs}=={(m,s) for m in ['ours','transformer'] for s in range(3)}
  assert all(p['config']['queries']==q and p['config']['steps']==10000 for p in parts)
  assert all(r['history'][-1]['step']==10000 and np.isfinite(r['fresh']['loss']) for r in rs)
  for name in ['ours','transformer']:
   a=[100*r['fresh']['accuracy'] for r in rs if r['model']==name]
   b=[100*r['all_queries_correct'] for r in rs if r['model']==name]
   rows.append(dict(task=task,model=name,parameters=9976 if name=='ours' else 9970,
       accuracy_mean=float(np.mean(a)),accuracy_sd=float(np.std(a,ddof=1)),all_correct_mean=float(np.mean(b))))
  rows.append(dict(task=task,model='frequency shortcut',accuracy_mean=float(100*(mode[:,None]==y[:,:q]).mean()),
                   all_correct_mean=float(100*(mode[:,None]==y[:,:q]).all(-1).mean())))
  all_runs.extend(dict(r,task=task) for r in rs)
 output=dict(protocol='MQAR-inspired causal synthetic task, not official MQAR benchmark',pairs=5,rows=rows,runs=all_runs)
 Path('associative-results.json').write_text(json.dumps(output,indent=2)+'\n')
 lines=['Five stored key/value pairs; single query versus three sequential queries.',
        'Ours: six hidden layers, 64 embedding dims, 12 units each, no residuals, 9,976 parameters.',
        'Transformer: one block, 16 dims, four heads, FF width 162, 9,970 parameters.',
        'Three seeds; 10,000 compiled Adam updates, batch 64, learning rate .001.',
        'Single query supervises one value/context; multi query supervises three values/context.',
        'Teacher-forced causal prefixes: no current or future answer is visible.',
        'Evaluation: same new 16,384 sequences for both models; mean +/- sample SD.',
        'This is an MQAR-style variant, not the official benchmark.','']
 for r in rows:
  lines.append(f"{r['task']} / {r['model']}: accuracy {r['accuracy_mean']:.2f}% +/- {r.get('accuracy_sd',0):.2f}; all answers correct {r['all_correct_mean']:.2f}%")
 text='\n'.join(lines)+'\n';Path('associative-summary.txt').write_text(text);print(text)
 with Path('README.md').open('a') as f:f.write('\n## Single- and multi-query associative recall\n\n'+text+'\nNeither model solves these settings: query accuracy is near the frequency\nshortcut. This does not establish architectural equivalence or a general\ntransformer limitation. The comparison uses a single attention block; deeper\nattention configurations have not been tested on this exact task. Raw outputs\nare in `associative-single-seed-*` and `mqar-seed-*`; `mqar_test.py` generates and\ntrains the tasks. The paper remains paused.\n')


if __name__=='__main__':main()
