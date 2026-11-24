# Last-token multiplication versus concatenation

The width-64 token-product baseline pools position × token vectors, then applies
the last-token embedding as a channelwise multiplier before normalization.
The ablation appends the last-token embedding to the same pooled memory before
normalization and the one-hidden-layer GELU readout. Both use Adam 0.0003,
batch size 64, identical training streams and held-out samples, and seeds 0–2.
Retrieval has 100,000 multiply and 99,964 concat parameters. Language has
81,153 multiply and 80,963 concat parameters. The hidden widths are 576 for multiply and 412 (retrieval) or 385 (language) for concat.

Accuracy is in percent. Values are mean ± sample standard deviation.

| Updates | Metric | Multiply | Concat |
|---:|---|---:|---:|
| 1,000 | Position copying | 59.89 ± 2.01 | 24.88 ± 0.37 |
| 1,000 | Single-query recall | 22.59 ± 2.05 | 30.48 ± 0.19 |
| 1,000 | Multi-query recall | 34.41 ± 0.46 | 34.85 ± 0.08 |
| 1,000 | Language accuracy | 28.18 ± 0.22 | 30.85 ± 0.31 |
| 1,000 | Language perplexity | 13.51 ± 0.15 | 11.35 ± 0.15 |
| 10,000 | Position copying | 98.86 ± 0.91 | 70.73 ± 1.20 |
| 10,000 | Single-query recall | 77.09 ± 5.28 | 30.03 ± 0.30 |
| 10,000 | Multi-query recall | 93.11 ± 2.92 | 34.82 ± 0.06 |
| 10,000 | Language accuracy | 40.35 ± 0.32 | 41.72 ± 0.20 |
| 10,000 | Language perplexity | 7.95 ± 0.18 | 7.25 ± 0.06 |

A seed-0 control gave concat 576 hidden units and 136,864 parameters.
- Position copying at 10k was 99.11% with multiplication, 70.10% with budget-matched concat, and 71.99% with 576-unit concat.
- Single-query recall at 10k was 80.38% with multiplication, 29.69% with budget-matched concat, and 29.78% with 576-unit concat.
- Multi-query recall at 10k was 91.58% with multiplication, 34.76% with budget-matched concat, and 34.66% with 576-unit concat.

## Query-swap diagnostic

At 10k updates, each single-query model saw 16,384 fixed memories
(sample seed 2026112401) queried once for each of their five stored keys.

| Metric | Multiply | Concat |
|---|---:|---:|
| Accuracy across all five queries | 76.93 ± 5.27% | 30.36 ± 0.16% |
| Memories with the same prediction for all five keys | 0.13 ± 0.10% | 46.90 ± 1.11% |
| Predictions among the most frequent stored values | 75.13 ± 1.08% | 97.30 ± 0.97% |

Multiplication lets the query directly rescale each pooled memory channel.
Concatenation gives the readout both inputs, so it must learn the selective
interaction. The query swaps test how strongly the trained model uses the key.
The wider control tests whether restoring the baseline hidden width recovers
retrieval at 10k. This experiment uses the raw width-64 pooling model.

## Reproduce

Run from the repository root. The first two commands use seeds 0–2.

```bash
out=experiments/last-token-concat-10k
for seed in 0 1 2; do
  python -m experiments.longer_training_100k --architecture token_product_last_concat --budget 100000 --steps 10000 --checkpoints 1000 10000 --seed "$seed" --tasks position single multi --output-prefix "$out/retrieval"
  python -m experiments.longer_training_100k --architecture token_product_last_concat --budget 81153 --steps 10000 --checkpoints 1000 10000 --seed "$seed" --tasks language --output-prefix "$out/language"
done
python -m experiments.longer_training_100k --architecture token_product_last_concat --steps 10000 --checkpoints 10000 --seed 0 --tasks position single multi --output-prefix "$out/same-hidden"
```

The existing multiply runs are in [the matched learning-rate comparison](../current64-lr0.0003-50k/comparison.txt).
Per-seed query-swap metrics are in [query-swap.json](query-swap.json).
