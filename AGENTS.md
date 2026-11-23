# Current model references

Use these configurations when referring to our current pooling model and
Transformer (TF) reference. Results below are means across three seeds;
± values are sample standard deviations. Check the linked result files when
new experiments are available.

## Our pooling model

Our strongest pooling model overall so far is `token_product`:
last-token-conditioned position–token pooling, embedding width 64, one GELU
hidden layer of width 576, and parameter-free pooled normalization.
Learned position and token embeddings interact by elementwise multiplication:

```text
pooled = sum(position[i] * token[i] * token[last] for i < last)
logits = Linear(GELU(Linear(normalize(pooled))))
```

The final token conditions the memory and is excluded from the sum. There are
100,000 parameters for retrieval tasks and 81,153 for language prediction.
The experimental implementation is `TokenInteraction` in
`experiments/last_token_probe.py`, with the `TokenProductModel` wrapper in
`experiments/longer_training_100k.py`. The reusable package default still uses
the earlier six-hidden-layer pooling model.

Final results after 10,000 updates per task:

| Task | Accuracy | Perplexity |
|---|---:|---:|
| Position copying | 99.55% ± 0.75 | — |
| Single-query associative recall | 95.21% ± 3.48 | — |
| Multi-query associative recall | 98.09% ± 1.16 | — |
| Next-character language prediction | 43.68% ± 0.33 | 7.58 ± 0.14 |

Source: [token-product100k-summary.json](experiments/token-product100k-summary.json).
The sigmoid and no-normalization follow-ups performed substantially worse on
recall; retain normalization and GELU as the current overall choice.
The conditioned model now has three-seed results at 1k/10k/30k/50k updates in
[the current comparison](experiments/current64-50k/summary.txt). At 50k, accuracy
is 99.99% ± 0.01 copying, 98.31% ± 0.91 single-query recall, 99.37% ± 0.30
multi-query recall, and 46.91% ± 0.27 language, with perplexity 7.61 ± 0.22.
CPU speed benchmarks against the current sinusoidal Transformer are in
[the speed comparison](experiments/current64-50k/speed-summary.txt). Seven warmed,
alternating trials on an Intel Core Ultra 7 155H show token-product speedups of
2.63–2.83x for batch-one inference, 2.71–4.54x for batch-64 inference, and
2.68–8.35x for compiled batch-64 Adam updates across the four tasks. These use
seed-0 50k checkpoints, resident inputs, and exclude compilation/data generation;
CPU implementation timings do not measure time to equal quality.

## Current Transformer reference

- Two causal Transformer blocks, embedding width 64, four attention heads.
- Feed-forward width 200 per block, GELU activation.
- Fixed sinusoidal position encodings (base 10,000), added to token embeddings; pre-LayerNorm with learned parameters, residual connections.
- 98,032 parameters for retrieval tasks; 94,033 for language prediction.
- Sinusoidal positions are the default going forward. Use `positions="learned"`
  in the model or `--positions learned` in longer training for historical reproduction.
- Three-seed multi-query recall at 10k: sinusoidal 100.00% ± 0.00,
  learned 78.80% ± 36.72. Source: `experiments/tf-positions-multi-10k-summary.json`.
- Other sinusoidal 10k results have only seed 0: copying 100%, single-query
  recall 28.16%, language accuracy 43.21%, perplexity 7.32. Source:
  `experiments/tf-positions-10k-seed-0.json`. These are now supplemented by
  three-seed results for all four tasks at 1k/10k/30k/50k in
  `experiments/current64-50k/summary.txt`. At 50k, sinusoidal accuracy is
  100.00% ± 0.00 copying, 29.93% ± 0.39 single-query recall, 100.00% ± 0.01
  multi-query recall, and 46.63% ± 0.42 language, with perplexity 6.79 ± 0.20.
- Implementation: `experiments/depth_100k_test.py`; longer training uses
  `experiments/longer_training_100k.py --architecture transformer`.
- Single-query failure diagnostic: changing only Adam's learning rate from
  0.001 to 0.0003 yields 100.00% ± 0.00 across three seeds at 10k updates on
  a separate 16,384-sequence test; the original 0.001 checkpoints score
  28.71% ± 0.22 on that same sample. The failed models largely ignore the
  queried key and choose frequent stored values. This is evidence of a
  training-regime failure, not a demonstrated Transformer capacity limit.
  The lower rate was selected exploratorily using seed 0 and replicated with
  seeds 1–2; existing reference tables retain Adam 0.001. See
  `experiments/single-query-investigation/summary.txt` and the reproducible
  diagnostic `experiments/investigate_single_query.py`.

The table below is the **historical learned-position** reference, not the new
sinusoidal default.

| Metric | 10k updates | 30k updates | 50k updates |
|---|---:|---:|---:|
| Position copying accuracy | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| Single-query recall accuracy | 28.77% ± 0.66 | 30.09% ± 0.31 | 29.99% ± 0.31 |
| Multi-query recall accuracy | 80.21% ± 34.18 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| Language accuracy | 43.16% ± 0.46 | 46.07% ± 0.16 | 46.72% ± 0.76 |
| Language perplexity ↓ | 7.20 ± 0.20 | 6.77 ± 0.05 | 6.81 ± 0.11 |

Source: [longer-transformer100k-summary.txt](experiments/longer-transformer100k-summary.txt).
The historical learned-position Transformer has the best language perplexity in this comparison at the
reported 30k milestone; the new conditioned-pooling and sinusoidal comparison at that duration is
reported separately in `experiments/current64-50k/summary.txt`.

## Shared training and evaluation context

Both references use Adam at learning rate 0.001, batch size 64, without a
learning-rate schedule or weight decay. Model and optimizer state remain
continuous across 1,000-update compiled chunks. Report predeclared final
checkpoints, without best-step selection.

Retrieval evaluation uses 16,384 held-out sequences. Multi-query accuracy scores
answers individually, with each prediction seeing only its causal prefix.
Language uses Tiny Shakespeare, a contiguous 80/10/10 split, context length 32,
and 4,096 test windows. Parameter budgets are approximate, and equal update
counts do not imply equal compute. These are small exploratory benchmarks;
avoid general architecture superiority claims.
