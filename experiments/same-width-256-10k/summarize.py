import json, statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'experiments/same-width-256-10k'
models = {
    'Learned-norm pooling': 'experiments/norm-query-learned-d256-50k/seed-seed-{}.json',
    'One-block Transformer': 'experiments/same-width-256-10k/tf1-seed-{}.json',
    'Two-block Transformer': 'experiments/same-width-256-10k/tf2-seed-{}.json',
}
lines = ['10,000 updates; three seeds; mean +/- sample standard deviation.',
         'Adam 0.0003, batch 64; held-out retrieval 16,384 sequences; language 4,096 windows.',
         'All models width 256; Transformers four heads, sinusoidal positions.',
         'Existing pooling runs reused at their 10k checkpoints.', '']
for name, pattern in models.items():
    rows = [json.loads((root / pattern.format(seed)).read_text()) for seed in range(3)]
    assert all(row['config']['lr'] == 0.0003 and row['config']['batch'] == 64 for row in rows)
    lines.append(name + ':')
    for task in ['position', 'single', 'multi', 'language']:
        records = [next(c['final'] for r in row['runs'] if r['task'] == task for c in r['checkpoints'] if c['step'] == 10000) for row in rows]
        accuracy = [100 * r['accuracy'] for r in records]
        result = f'  {task}: {statistics.mean(accuracy):.2f}% +/- {statistics.stdev(accuracy):.2f}'
        if task == 'language':
            ppl = [r['perplexity'] for r in records]
            result += f', perplexity {statistics.mean(ppl):.2f} +/- {statistics.stdev(ppl):.2f}'
        lines.append(result)
    lines.append('')
text = '\n'.join(lines).rstrip() + '\n'
(out / 'summary.txt').write_text(text)
print(text)
