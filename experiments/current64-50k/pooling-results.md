Pooling results: width 64; 100,000 retrieval / 81,153 language parameters.
Three seeds; means +/- sample standard deviation. Both models have completed all tasks and checkpoints.

| Metric | 1k | 10k | 30k | 50k |
|---|---:|---:|---:|---:|
| Copy accuracy (%) | 75.52 ± 0.53 | 99.55 ± 0.75 | 100.00 ± 0.00 | 99.99 ± 0.01 |
| Single-query accuracy (%) | 31.17 ± 0.71 | 95.21 ± 3.48 | 98.08 ± 0.98 | 98.31 ± 0.91 |
| Multi-query accuracy (%) | 37.02 ± 3.05 | 98.09 ± 1.16 | 99.17 ± 0.39 | 99.37 ± 0.30 |
| Language accuracy (%) | 34.62 ± 0.40 | 43.68 ± 0.33 | 46.29 ± 0.21 | 46.91 ± 0.27 |
| Language perplexity | 10.14 ± 0.15 | 7.58 ± 0.14 | 7.52 ± 0.11 | 7.61 ± 0.22 |

The completed comparison of both models is in summary.json and summary.txt.
Run from repository root: python experiments/run_current64_50k.py
Summarize completed runs: python experiments/run_current64_50k.py --summarize-only

Same Adam settings, fixed held-out samples, and continuous training state as the current references. Equal update counts do not imply equal compute.
