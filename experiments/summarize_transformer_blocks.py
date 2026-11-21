"""Aggregate 1/2/4-block budget-matched Transformer comparison."""
import json
from pathlib import Path
import numpy as np
rows=[];runs=[];configs={}
for arch in ['one','two','four']:
 for seed in range(3):
  part=json.loads(Path(f'tfblocks-{arch}-seed-{seed}.json').read_text());configs[arch]=part['config']
  assert len(part['runs'])==4
  for r in part['runs']:
   assert r['parameters']<=10000 and r['history'][-1]['step']==10000 and np.isfinite(r['final']['loss'])
   runs.append(dict(r,architecture=arch))
 for task in ['position','single','multi','language']:
  rs=[r for r in runs if r['architecture']==arch and r['task']==task]
  acc=[100*r['final']['accuracy'] for r in rs];ppl=[r['final']['perplexity'] for r in rs]
  rows.append(dict(architecture=arch,blocks=configs[arch]['blocks'],task=task,parameters=rs[0]['parameters'],ff_hidden=configs[arch]['hidden_by_task'][task],accuracy_mean=float(np.mean(acc)),accuracy_sd=float(np.std(acc,ddof=1)),perplexity_mean=float(np.mean(ppl)),perplexity_sd=float(np.std(ppl,ddof=1))))
Path('tfblocks-results.json').write_text(json.dumps(dict(configs=configs,rows=rows,runs=runs),indent=2)+'\n')
lines=['Transformer block-count comparison at <=10k total trainable parameters.', 'd16, four heads, learned positions, one standard FF hidden layer per block.', 'Three seeds, 10,000 compiled Adam updates, batch64, lr .001.', 'Same initial seed and training/evaluation examples; FF widths adjusted to budget.', 'All preceding blocks compute every state; final block computes only supervised outputs.', 'Mean +/- sample standard deviation. No speed benchmark.','']
for task in ['position','single','multi','language']:
 for r in [v for v in rows if v['task']==task]:
  lines.append(f"{task} / {r['blocks']} blocks: {r['parameters']} params, FF {r['ff_hidden']}, accuracy {r['accuracy_mean']:.2f}% +/- {r['accuracy_sd']:.2f}"+(f", perplexity {r['perplexity_mean']:.2f} +/- {r['perplexity_sd']:.2f}" if task=='language' else ''))
text='\n'.join(lines)+'\n';Path('tfblocks-summary.txt').write_text(text);print(text)
with (Path(__file__).resolve().parents[1] / 'README.md').open('a') as f:f.write('\n## One versus multiple Transformer blocks, same small budget\n\n'+text+'\nReproduce with `transformer_blocks_test.py --architecture one|two|four --seed 0|1|2`, then `summarize_transformer_blocks.py`. Every trainable embedding, norm, attention, FFN, and output parameter is counted. Context32 for language;16 for query tasks, using the same generators and evaluation samples as `benchmark_100k.py`. Optimized final-block logits/gradients and five-step full compiled model/optimizer-state parity are checked in `check_transformer_blocks.py`. Fixed parameters imply narrower FFNs at greater depths, so this tests budget allocation rather than adding blocks at constant width throughout. Previous main one-block reference remains unchanged; no paper edits made.\n')
