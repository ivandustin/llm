# Last-Token-Conditioned Pooling for Associative Recall and Language Prediction

Flax NNX implementations and reproducible comparisons of last-token-conditioned
position–token pooling and small causal Transformers on position copying,
associative recall, and Tiny Shakespeare next-character prediction. The paper
describes the models, experiments, and limitations. [Paper (PDF)](paper/paper.pdf)

## Matched 50,000-update results

All models used Adam at **0.0003** (the default for longer training), batch
size 64, and three seeds. Values are means ± sample standard deviations at the
final checkpoint. Retrieval uses 16,384 held-out sequences, with multi-query recall
scored per answer on its causal prefix. Language uses 4,096 Tiny Shakespeare
test windows at context length 32 from a contiguous 80/10/10 split.

| Metric | Token-product pooling (64) | Concat + GELU pooling (128) | Sinusoidal Transformer (64) |
|---|---:|---:|---:|
| Position copying accuracy | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| Single-query recall accuracy | 98.33% ± 1.27 | 99.98% ± 0.02 | 100.00% ± 0.00 |
| Multi-query recall accuracy | 99.17% ± 0.66 | 99.98% ± 0.02 | 100.00% ± 0.00 |
| Next-character accuracy | 45.67% ± 0.41 | 47.00% ± 0.22 | 46.63% ± 0.72 |
| Language perplexity ↓ | 7.20 ± 0.20 | 7.09 ± 0.07 | 6.69 ± 0.12 |
| Parameters (retrieval / language) | 100,000 / 81,153 | 99,858 / 99,743 | 98,032 / 94,033 |

The wider pooling variant (`token_product_raw_gelu`) concatenates the sum of
raw position–token–query products and the sum of their elementwise GELU
transforms. It applies parameter-free normalization and a one-hidden-layer
GELU readout to the concatenated vector. The run uses a 100,000-parameter
ceiling per task. At 50k it has the highest mean next-character accuracy,
and the Transformer has the lowest mean perplexity and reaches 100% on all
three retrieval tasks. Equal update counts do not imply equal compute.
[Wider pooling results](experiments/raw-gelu-all-50k/summary.txt) ·
[Matched width-64 results](experiments/current64-lr0.0003-50k/summary.txt)

## Width-256 additive raw + GELU

The additive variant (`token_product_raw_gelu_add`) pools
`sum(x) + sum(GELU(x))`, where `x` is the last-token-conditioned
position–token product at each preceding position. Parameter-free
normalization and a one-hidden-layer GELU readout follow pooling. The model
has 99,721 retrieval parameters and 99,923 language parameters. It uses the
same Adam settings, three seeds, and held-out evaluation as the comparison
above.

| Metric | 10k updates | 30k updates |
|---|---:|---:|
| Position copying accuracy | 99.99% ± 0.01 | 99.99% ± 0.00 |
| Single-query recall accuracy | 98.33% ± 1.12 | 99.99% ± 0.01 |
| Multi-query recall accuracy | 99.90% ± 0.01 | 100.00% ± 0.00 |
| Next-character accuracy | 43.29% ± 0.47 | 46.29% ± 0.51 |
| Language perplexity ↓ | 7.32 ± 0.11 | 6.98 ± 0.14 |

This is our strongest overall pooling result for retrieval at 30k. The
multi-query mean rounds to 100.00% with one missed answer across the three
seeds. Width-128 concatenated GELU still has the highest three-seed mean
next-character accuracy at 50k. The different update counts leave the
language comparison open.

From the repository root, reproduce both checkpoints for seeds 0–2.

```bash
for seed in 0 1 2; do
  python -m experiments.longer_training_100k \
    --architecture token_product_raw_gelu_add --width 256 --budget 100000 \
    --lr 0.0003 --steps 30000 --checkpoints 10000 30000 --seed "$seed"
done
```

## Width-64 learning-rate comparison

The earlier [Adam 0.001 comparison](experiments/current64-50k/summary.txt)
covers the same width-64 models and all four tasks through 50k updates. At
50k, Transformer single-query recall rose from 29.93% ± 0.39 at 0.001 to
100.00% ± 0.00 at 0.0003. Width-64 pooling learned more slowly early on at
the lower rate. See the [learning-rate comparison](experiments/current64-lr0.0003-50k/comparison.txt).

## Short-budget language context screen

At 3,000 updates, concat + GELU pooling had the lowest mean test perplexity
at both context lengths. Each configuration used three seeds and Adam at
0.001, selected on a separate validation seed. This screen uses matched
target positions across context lengths and a separate sampling protocol
from the 50k comparison.

| Context length | Token-product pooling | Concat + GELU pooling | Sinusoidal Transformer |
|---|---:|---:|---:|
| 32 | 8.21 ± 0.05 | 7.92 ± 0.02 | 8.28 ± 0.19 |
| 128 | 8.94 ± 0.12 | 8.35 ± 0.11 | 8.60 ± 0.40 |

The [context screen summary](experiments/language-context-screen/summary.txt)
includes test accuracy, training times, and batch-one inference latency.

## Width-64 CPU timings

On an Intel Core Ultra 7 155H, token-product pooling ran 2.63–2.83× as fast
as the sinusoidal Transformer for batch-one inference, 2.71–4.54× for
batch-64 inference, and 2.68–8.35× for compiled batch-64 Adam updates across
the four tasks. These are medians from seven alternating warmed JAX trials
using seed-0 50k checkpoints from the Adam 0.001 comparison and resident
inputs. Timings exclude compilation, input generation, and transfers.
Time to reach the same quality also depends
on how many updates each model needs. See the
[CPU benchmark](experiments/current64-50k/speed-summary.txt).

## Width-64 token-product pooling

Token-product pooling multiplies each previous position–token vector by the final
token's embedding, then sums, normalizes, and predicts with one GELU hidden
layer. The final token supplies the conditioning vector and is excluded from
the sum. The same rule applies to all four tasks.

```text
pooled = sum(position[i] * token[i] * token[last] for i < last)
logits = Linear(GELU(Linear(normalize(pooled))))
```

Here `*` is elementwise multiplication. Algebraically, the final-token product
can move outside the sum, requiring only 64 extra multiplications per example
at embedding width 64. The experiment implements the per-token form above;
the factored form can differ slightly in floating-point rounding. No activation
is applied before summation, and normalization has no learned scale or bias.

After **10,000 updates per task**, batch size 64, and three seeds:

| Task | Parameters | Accuracy, mean ± sample SD |
|---|---:|---:|
| Position copying | 100,000 | 99.55% ± 0.75 |
| Single-query associative recall | 100,000 | 95.21% ± 3.48 |
| Multi-query associative recall | 100,000 | 98.09% ± 1.16 |
| Next-character language prediction | 81,153 | 43.68% ± 0.33 |

Language perplexity is **7.58 ± 0.14**. Dimensions are embedding width 64 and
one hidden layer of width 576. Each task starts from scratch with Adam at 0.001,
no schedule or weight decay, and continuous model/optimizer state across
1,000-update scans. These are final 10k checkpoints, without best-step selection.
Only this pooled model was newly trained in the four-task follow-up.

Retrieval uses 16,384 held-out sequences. Single recall asks for one stored
key's value; multi recall asks for three distinct keys' values, scoring answers
individually. Each prediction sees only its causal prefix, including correct
earlier answers but excluding current and future answers. Language retains the
Tiny Shakespeare contiguous 80/10/10 split, context 32, and 4,096 test windows.

A separate 3,000-update multi-query diagnostic used identical initialization,
data, and exactly 100,000 parameters for each variant:

| Pooling rule | Accuracy, mean ± sample SD |
|---|---:|
| Previous tokens only, without final-token interaction | 34.24% ± 0.28 |
| Multiply by final-token embedding | 88.25% ± 0.93 |
| Multiply, then GELU per token before summing | 87.93% ± 1.62 |

All three diagnostic variants exclude the final token from the summed memory.
The multiplication variant gets all three answers correct in 68.74% of
sequences, distinct from its per-answer accuracy. Earlier experiments that
concatenated the final-token embedding with the pool showed no short-run gain.

See the [10k summary](experiments/token-product100k-summary.json), raw runs for
[seed 0](experiments/token-product100k-seed-0.json),
[seed 1](experiments/token-product100k-seed-1.json), and
[seed 2](experiments/token-product100k-seed-2.json), and the
[3k diagnostic](experiments/token-interaction-shallow-probe.json).

## Install

Python 3.11 or newer is required. From a fresh checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

`pyproject.toml` and `requirements-tested.txt` pin the tested CPU dependencies.
Alternatively, install them with `python -m pip install -r requirements.txt`.
The experiments run on CPU with JAX JIT compilation. A GPU is not required.

## Repository layout

- `position_token_pooling/`: reusable model package.
- `experiments/`: experiment scripts, saved results, summaries, logs, and checkpoints.
- `tests/`: package validation.
- `data/`: input datasets.
- `paper/`: paper source and PDF.

## Reusable package model

The original package default is learned position/token embeddings, elementwise product,
sum over positions, parameter-free normalization, six GELU hidden layers, and
an output linear layer. It has no residuals, query gate, or final normalization.
The original single-hidden-layer sigmoid baseline is also retained. The
last-token-conditioned pooling variants are implemented in
[`TokenInteraction`](experiments/last_token_probe.py), with the standard-task
wrapper `TokenProductModel` in
[`longer_training_100k.py`](experiments/longer_training_100k.py). The package
default and the original-model example below retain their earlier behavior.

```python
import jax.numpy as jnp
from flax import nnx
from position_token_pooling import StackedModel

context, vocab = 16, 96
model = StackedModel([context, vocab], vocab, 64, 120, nnx.Rngs(0),
                     hidden_layers=6, residual=False)
tokens = jnp.arange(context) % 16
example = jnp.stack([jnp.arange(context), tokens])
logits = model(example)  # shape (vocab,)
```

`position_token_pooling/` contains the reusable model implementations. The scripts in `experiments/` contain task generators, Transformer implementations, training, evaluation,
and timing tools. The compatibility imports `candidate_model.py` and
`stacked_candidate_model.py` keep earlier experiment commands working. Parameter
names and initialization order are preserved for existing `.npz` checkpoints.

## Current Transformer positional encoding

The two-block Transformer defaults to fixed sinusoidal positions (base 10,000),
added to token embeddings. It uses embedding width 64, four attention heads, a
width-200 GELU feed-forward layer per block, learned pre-LayerNorm, and residual
connections. It has 98,032 trainable parameters for retrieval and 94,033 for
language. Pooling uses learned positions.

At 10,000 updates, multi-query recall across seeds 0–2 is **100.00% ± 0.00**
with sinusoidal positions versus **78.80% ± 36.72** with learned positions
(sample standard deviations). See [results](experiments/tf-positions-multi-10k-summary.json).
The [matched width-64 results](experiments/current64-lr0.0003-50k/summary.txt)
cover all four tasks at 1k, 10k, 30k, and 50k updates across three seeds.

`longer_training_100k.py --architecture transformer` uses sinusoidal positions
and writes `longer-transformer-sinusoidal100k-lr0.0003-seed-*` artifacts by
default. For historical learned-position training, use
`--positions learned --lr 0.001` from `experiments/`, where the script can find
its reference files. The standalone 10k `depth_100k_test.py` Transformer run
uses Adam 0.001 and writes `depth100k-sinusoidal-transformer-seed-*` artifacts.

## Reproduce experiments

Install the project from the repository root with `python -m pip install -e .`.
Run the following commands from the root. Tiny Shakespeare is included at
`data/shakespeare.txt`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Matched width-64 comparison at 1k, 10k, 30k, and 50k updates.
python -m experiments.run_current64_50k --lr 0.0003

# Width-128 concat + GELU comparison.
for seed in 0 1 2; do
  python -m experiments.longer_training_100k --architecture token_product_raw_gelu --width 128 --hidden-layers 1 --budget 100000 --lr 0.0003 --steps 50000 --checkpoints 10000 30000 50000 --seed "$seed" --output-prefix experiments/raw-gelu-all-50k/d128-n1
done

# Short-budget language context screen.
JAX_PLATFORMS=cpu python -m experiments.language_context_compare --output experiments/language-context-screen/results.json
```

These commands write to the published experiment paths. Use another output
path to keep a reproduction alongside the checked-in results. The following
historical 10k commands also start at the repository root, then switch to
`experiments/` to write artifacts there. Run them after the editable install.

```sh
cd experiments

# Last-token-conditioned pooling: four tasks, 10k updates, three seeds.
for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.longer_training_100k --architecture token_product --steps 10000 --lr 0.001 --seed "$seed"
done

# 16-dimensional token-product pooling, <=10k parameters per task.
for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.longer_training_100k --architecture token_product --width 16 --budget 10000 --steps 10000 --lr 0.001 --seed "$seed" --output-prefix token-product-d16-10k
done

# 16-dimensional two-block sinusoidal Transformer, <=10k parameters per task.
for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.longer_training_100k --architecture transformer --width 16 --budget 10000 --steps 10000 --lr 0.001 --seed "$seed" --output-prefix transformer-d16-10k
done

# Six-layer pool and sinusoidal Transformer at Adam 0.001, four tasks, one seed.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.benchmark_100k --architecture ours --seed 0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m experiments.depth_100k_test --architecture transformer --seed 0
```

Repeat with seeds 1 and 2 for the reported three-seed protocol. These commands
perform 10,000 updates per task and write result JSON and checkpoint files.
The first script also retains a historical one-block Transformer option; the
current reference uses two standard blocks. Wider, deeper, residual, and parallel
branch experiments have their own scripts documented in the experiment log below.

The conditioned runs write `token-product100k-seed-<seed>.json` and checkpoint
directories of the same name in the current directory. To repeat the controlled
3k multi-query diagnostic from `experiments/`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.last_token_probe --layers 1 --hidden 576 --steps 3000 --seeds 3 --variants previous_pool token_product token_product_gelu --output token-interaction-shallow-probe.json
```

Training loops compile chunks of updates and preserve model/Adam state between
chunks. Data preparation, evaluation, and saving are outside those compiled loops.
Saved `.npz` files contain model weights, not optimizer state for exact resumption.

The paper is [paper/paper.pdf](paper/paper.pdf), with source in
[paper/paper.tex](paper/paper.tex). Rebuild it with `pdflatex paper.tex` twice from
`paper/` (a separate LaTeX installation is needed). Result tables are exploratory,
not evidence that this model generally replaces attention.

## Historical experiment log

The sections below preserve the configurations, controls, and results in the
order they were investigated. Older one-block references remain explicitly
historical; see the two-block and approximately 100k comparisons for later controls.
Statements about weak associative recall or a paused paper describe those
historical configurations and stages. The latest conditioned-pooling results
and current paper are linked above.

### Original language benchmark

The original sigmoid baseline is implemented locally in `position_token_pooling/base.py`:

`position indices, token indices → learned embeddings → elementwise product → sum over positions → Linear → sigmoid → Linear`

The comparison is a one-block, four-head causal transformer with width 32,
learned token and position embeddings, pre-layer normalization, residuals,
a GELU feed-forward layer of width 128, and a final vocabulary projection.
The transformer uses its last position to predict the next character.

Both models receive identical batches in each seed and train with Adam at 0.001,
batch size 64, context length 32, and 10,000 updates. Each update supervises one
next character per context for both models. The cube uses embedding width 32
and hidden width 152: 18,065 parameters versus the transformer's 18,017.
No dropout, weight decay, learning-rate search, or checkpoint selection is used.
Final checkpoints are evaluated over the same 4,096 sampled windows in each
held-out split. Reported seed variation is across three training seeds;
evaluation windows are shared across seeds and may overlap.

Data: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt),
1,115,394 characters and 65 character types. Contiguous 80%/10%/10% training,
validation, and test splits; context windows do not cross split boundaries.
Vocabulary is enumerated from the complete text; no held-out frequencies or
labels enter training. Data checksum and exact split lengths are in results.json.
This is a character-level feasibility test, not a word-token benchmark.

Run from `experiments/` after installing the project dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmark.py --steps 10000 --lr 0.001 --seeds 3 > run.log 2>&1
```

`results.json` contains metrics, parameter counts, training timings and generated
samples. `pilot-results.json` preserves the initial 1,000-update run.
`requirements-tested.txt` records installed dependency versions.

Training timings exclude the first update (compilation), held-out evaluation,
and text generation. They include batch sampling, Python dispatch, and waiting
for each step's loss. Compare them as local CPU measurements, not hardware-independent
speed claims. Equal parameters and updates do not imply equal compute; the
transformer processes every position internally. A larger dataset, longer training,
and tuning for each model would be needed for a general architecture comparison.

Runtime checks confirmed that reversing tokens changes the pooling model predictions
and that vectorized batch results equal independently evaluated examples.
Standalone regression checks are in `tests/test_standalone.py`.

#### Results

Means ± sample standard deviations across three seeds.

| Model | Parameters | Test perplexity ↓ | Test accuracy ↑ | CPU training seconds ↓ |
|---|---:|---:|---:|---:|
| cube | 18,065 | 8.19 ± 0.07 | 38.12 ± 0.27% | 11.15 ± 0.24 |
| transformer | 18,017 | 7.92 ± 0.16 | 40.84 ± 0.14% | 59.39 ± 4.35 |

## Controlled capability tests

`capabilities.py` tests the original local pooling model against a two-block causal
transformer. These probes are separate from the character-language benchmark
above. The main probe suite uses embedding width **16** for both models (the
character benchmark used 32), four attention heads, batch size 64, Adam at
0.001, and 3,000 updates. The cube's hidden width is selected to match the
transformer's total parameter count within eight parameters: 10,024 versus
10,016 at context 16, and 10,792 versus 10,784 at context 64.

Each context length is trained independently, with seeds 0, 1, and 2. Models
receive identical generated training batches within a task and seed. Each task
has 4,096 fixed evaluation examples, independent of the training RNG. The same
evaluation examples are shared across models and seeds. Evaluation is monitored
at updates 500, 1,500, and 3,000; results use the final update, without checkpoint
selection. A vocabulary of 96 input and output symbols is shared by every task;
most tasks use only a subset of output classes.

| Task | Construction | Uniform valid-label chance |
|---|---|---:|
| First-token copy | Return the first of L independent 16-way symbols | 6.25% |
| Queried-position copy | Final symbol encodes which of the preceding L−1 positions to return | 6.25% |
| Associative recall | Alternating unique keys and independent 16-way values; final symbol queries one key | 6.25% |
| Relative order | Return whether the unique A precedes the unique B among distractors | 50% |
| Majority | Return the majority bit among L−1 bits; final position is a dummy | 50% |

There are 7 key–value pairs at context 16 and 31 at context 64. Keys are drawn
without replacement from 32 key types, values from 16 separate symbol types;
a dummy fills any unused position. Majority inputs contain an odd number of
bits, eliminating ties. Training uses fresh random samples, but support is not
explicitly partitioned: in particular the 15-bit majority task has only 32,768
possible inputs, so evaluation patterns can also occur in training. The 63-bit
majority task offers a much larger support. These are controlled task probes,
not tests of natural-language reasoning or length extrapolation.

The frequency baseline predicts the most common candidate token, breaking ties
by smallest symbol ID. It ignores the query on queried-copy and recall tasks.
For relative order the comparison baseline always predicts class zero. Frequency
is an exact algorithm for the majority task. It is a stronger shortcut baseline
than uniform guessing on copy and recall because the target is itself among
the candidates.

Additional controls, all at context 16 and across three seeds:

- `capabilities-longer-training.json`: cube width 16, queried copying, 30,000 updates.
- `capabilities-retrieval-extended.json`: both models, width 16, associative recall, 10,000 updates.
- `capabilities-wider.json`: cube width **64**, queried copying, 10,000 updates; hidden width and total parameters also increase (113,524 parameters). This is a capacity control, not a parameter-matched comparison against the smaller baseline.

Reproduce serially:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python capabilities.py > capabilities.log 2>&1
python capabilities.py --models cube --tasks indexed_copy --contexts 16 --steps 30000 --output capabilities-longer-training.json
python capabilities.py --tasks associative_recall --contexts 16 --steps 10000 --output capabilities-retrieval-extended.json
python capabilities.py --models cube --tasks indexed_copy --contexts 16 --width 64 --steps 10000 --output capabilities-wider.json
python summarize_capabilities.py
```

The actual main run completed context 16 serially, then resumed independent
context-64 seeds in three processes, with CPU affinities 0–5, 6–11, and 12–17.
`capabilities-before-parallel.json` preserves completed work before rescheduling;
`capabilities-part-{0,1,2}.json` records the remaining seed runs. The final
`capabilities-results.json` combines completed runs without duplicates. Some
controls ran concurrently. Synthetic training times therefore should not be
used for a speed comparison. The original isolated character benchmark timings
above are unaffected.

`capabilities-summary.txt` reports final accuracies and between-seed standard
deviations. `summarize_capabilities.py` regenerates the summary and frequency
baselines directly from the fixed evaluation generators. Generator checks verify
labels and valid symbol ranges for every task at lengths 16 and 64. Failure to
learn a probe at these settings is not a mathematical impossibility result;
other widths, optimizers, initializations, or training budgets may change it.

## Information preservation and representational capacity

`python information_check.py` runs two **hand-designed**, untrained checks using
the local pooling modules. Their results are in `information-check.json`.

First, set embedding width d=L, position vector P[i] to the i-th unit vector,
and token vector E[t] to the scalar token ID t repeated across all L coordinates.
Then `sum_i P[i] * E[x_i]` is exactly `[x_0, ..., x_(L-1)]`. This preserves the
entire sequence of IDs. Checks at L=16 and L=64 each recover 1,024 random
sequences exactly, with zero floating-point error. This is a sufficient
construction, not a lower bound on necessary width or a claim about the
embeddings learned in the benchmarks.

Second, a constructed full the base pooling model uses position-specific one-hot blocks
(d=L*V=1,536), and one hidden unit for each position/data-token pair
(h=15*16=240). Each sigmoid unit detects the conjunction of its position being
queried and its data token being present there. The output layer collects units
by data token. All embeddings, kernels, and biases are explicitly assigned.
This 564,048-parameter construction returns the queried token on all 4,096
fixed short-context evaluation examples (100%). It is neither trained nor
parameter matched, so it is not a competitive empirical result.

These checks establish that multiplication followed by summation need not
lose sequence information, and that the model family can represent queried
copying for bounded contexts. They do not show that randomly initialized,
smaller models learn the necessary encoding and readout. A training failure
cannot by itself distinguish loss of information in learned pooling from
failure to decode information that remains present. Establishing that would
require additional frozen-representation probes or other diagnostics.

## One-block transformer control

The main capability comparison matches total parameters but uses **two**
transformer blocks versus the cube's single sigmoid feed-forward hidden layer.
The original character-language benchmark uses **one** transformer block.
A transformer block includes attention and a feed-forward sublayer; block count
is not the same as counting individual linear layers.

To check the block-depth concern, `--depth 1` uses one transformer block with
embedding width 16 and expands that block's feed-forward hidden width to 164.
This matches the original probe cube within 12 parameters: 10,036 versus 10,024
at length 16, and 10,804 versus 10,792 at length 64. The cube is unchanged.
The control tests queried-position copying at both lengths, 3,000 updates,
three seeds, identical training batches, and the same evaluation examples.

```sh
python capabilities.py --models transformer --tasks indexed_copy --contexts 16 64 --depth 1 --output capabilities-one-block.json
```

The actual control runs independent seeds in parallel; the raw seed files are
`capabilities-one-block-{0,1,2}.json`. `capabilities-one-block.json` combines them.
These experiments and follow-up choices are exploratory; they are not a
preregistered or blind evaluation, and no comprehensive hyperparameter search
was performed for either architecture.

## Capability results

Final accuracy (%), mean across three seeds. The main transformer has two blocks.

| Task | Context | Cube | Transformer | Frequency baseline |
|---|---:|---:|---:|---:|
| first_copy | 16 | 100.00 | 100.00 | 20.65 |
| indexed_copy | 16 | 19.51 | 100.00 | 19.73 |
| associative_recall | 16 | 27.26 | 24.76 | 27.56 |
| relative_order | 16 | 100.00 | 100.00 | 49.29 |
| majority | 16 | 100.00 | 100.00 | 100.00 |
| first_copy | 64 | 100.00 | 100.00 | 12.35 |
| indexed_copy | 64 | 9.99 | 90.95 | 12.48 |
| associative_recall | 64 | 14.03 | 12.57 | 15.36 |
| relative_order | 64 | 99.81 | 100.00 | 50.00 |
| majority | 64 | 100.00 | 100.00 | 100.00 |

The one-block, parameter-matched transformer reaches 100% queried-copy accuracy
at context 16 and 96.39% at context 64; corresponding cube results are 19.51%
and 9.99%. Thus the observed queried-copy gap does not require two transformer
blocks. The wider cube averages 18.68% on short queried copying; the longer-trained
small cube averages 19.14%. Neither model solves associative recall at these
settings, including the 10,000-update short-context control. These observations
describe learnability under the tested configurations, not a proof of what the
model family can or cannot express.

## Encoding and readout diagnosis

`readout_experiments.py` separates encoding from readout learning on queried
copying at context 16. The baseline diagnostic has 255 embedding dimensions,
a 240-unit hidden layer, and 113,136 total parameters. This is larger than the
main 10,024-parameter capability model, so these diagnostic results are not
parameter-matched comparisons against the 10,016-parameter transformer.
The 64-dimension widened cube from the earlier control has a similar total
parameter count (113,524), but a different allocation between embeddings and
feed-forward width.

For the fixed-encoding variants, each of the 15 data positions gets its own
16-way one-hot block; a separate 15-way block records the query. Position
embeddings select their respective blocks and token embeddings populate them.
This uses the actual the base pooling model embedding and pooling operations. It is lossless
on this task's input support, verified by reconstructing every token of all
4,096 evaluation inputs. These embeddings encode observed inputs only, not the
answer. Freezing is verified by exact equality before and after training.

All variants train on identical fresh batches within a seed for 10,000 Adam
updates at 0.001, batch size 64. Evaluation uses a new fixed sample of 4,096
examples (RNG seed `700000 + context`), not the sample used to choose follow-up
experiments in the earlier suite. However, this is an exploratory study, not a
blind or preregistered evaluation. Three seeds run independently in parallel;
wall-clock timings are not comparable speed measurements.

The fixed variants change one readout choice from the sigmoid baseline:
parameter-free layer normalization before the first affine layer, GELU instead
of sigmoid, or an additional sigmoid hidden layer. The two-hidden-layer control
uses hidden width 164 to match the one-hidden-layer readout budget within 0.4%.
Thus its layer widths differ, rather than adding an equal-width layer and
increasing the budget. Normalization uses `(z - mean(z))/sqrt(var(z)+1e-5)`.

The random-initialization variants learn embeddings from the start and separately
test normalization, GELU, and their combination. A lossless-initialization sigmoid
control also learns embeddings from the start. All but the deeper control share
the same initial readout parameters within a seed. The diagnostic uses the
the original affine layers; the original sigmoid baseline is unchanged.

Reproduce the full diagnostic serially:

```sh
python readout_experiments.py --output readout-results.json
```

The actual runs use `readout-seed-{0,1,2}.json` and preserve `.npz` parameter
snapshots in corresponding directories. `summarize_readout.py` merges the seed
files and produces `readout-summary.txt`. The initial 3,000-update, one-seed
pilot is preserved as `readout-pilot.json`. A separate one-seed width-16,
hidden-72 check is in `readout-small-pilot.json`; it must not be interpreted
as a three-seed result.

A successful frozen readout (accuracy at least 99%) gets a paired continuation:
clone the same learned model twice, reset Adam to 0.0001 in both branches,
then train identical batches for 3,000 updates with embeddings either still
frozen or newly trainable. No checkpoint is chosen by maximum evaluation score;
reported main accuracies use the final update. The success threshold controls
which diagnostic receives a continuation and should be considered exploratory.

| Variant | Accuracy, mean ± sample SD (%) |
|---|---:|
| Fixed lossless encoding, sigmoid | 19.89 ± 0.05 |
| Fixed lossless encoding, normalization + sigmoid | 56.38 ± 0.45 |
| Fixed lossless encoding, GELU | 99.19 ± 0.09 |
| Fixed lossless encoding, two sigmoid hidden layers | 19.57 ± 0.42 |
| Lossless initialization, sigmoid, learn embeddings from start | 50.35 ± 7.97 |
| Random initialization, sigmoid | 18.59 ± 0.56 |
| Random initialization, normalization + sigmoid | 58.27 ± 1.73 |
| Random initialization, GELU | 41.38 ± 4.35 |
| Random initialization, normalization + GELU | 98.02 ± 0.35 |

After the paired continuation, fixed-encoding GELU reaches 99.64 ± 0.11% with
embeddings still frozen, and 99.76 ± 0.10% with embeddings unfrozen. The small
width-16, hidden-72 normalized-GELU pilot reaches only 19.70% (one seed).
This establishes a successful learning recipe for the larger short-context
probe; it does not establish that the smaller model or longer contexts are solved.

Activation diagnostics measure the first affine layer's input to its nonlinearity.
The initial frozen sigmoid readout has preactivation SD about 0.25 and zero
measured values with absolute magnitude above 4. The random-embedding baseline
has initial preactivation SD about 0.016. These observations do not support
initial sigmoid saturation in these conditions; the sigmoid is near its linear
region. GELU and normalization are supported by the controlled results, but the
experiments do not prove a single causal mechanism for their advantage. The
recorded sigmoid-derivative diagnostic is a hypothetical reference for GELU
variants, not the derivative of GELU itself.

## Experimental model variant

`candidate_model.Model` inherits the local base model constructor and
input convention. It keeps learned position/token embeddings, multiplication,
summation, and the original affine layers, while applying parameter-free layer
normalization to the pooled vector and GELU at the hidden layer. It adds no
parameters. A runtime check confirms its outputs match the successful diagnostic
implementation at identical initialization. The original sigmoid baseline is retained separately.

```python
from flax import nnx
from candidate_model import Model

model = Model([16, 96], 96, 255, 240, nnx.Rngs(0))
# Per-example input: stack([position_indices, token_indices]), shape (2, 16).
# For batches, use nnx.vmap(model), as with the original base model.
```

The original character-language benchmark also accepts
`--cube-variant norm_gelu --models cube --output candidate-language.json`.
The candidate language check uses the original width 32, hidden width 152,
18,065 parameters, training batches, held-out windows, and 10,000-update budget.
Its concurrent runs are not used for a new speed comparison.

## Candidate language results

Means ± sample standard deviations across three seeds; original models use
`results.json`, candidate uses `candidate-language.json`.

| Model | Parameters | Test perplexity | Test accuracy (%) |
|---|---:|---:|---:|
| Original cube | 18,065 | 8.19 ± 0.07 | 38.12 ± 0.27 |
| Normalized GELU cube | 18,065 | 7.81 ± 0.12 | 40.36 ± 0.25 |
| Transformer | 18,017 | 7.92 ± 0.16 | 40.84 ± 0.14 |

The candidate improves on the original cube at unchanged parameter count and
lands near the transformer on this small language benchmark. This is not a
claim of general transformer equivalence. The large queried-copy improvement
uses width 255 and 113,136 parameters; the small width-16 pilot remains weak.
The language check uses width 32 and 18,065 parameters. Synthetic and language
results must not be conflated, and concurrent timings do not establish a speed
advantage for the candidate.

## Strict 10,000-parameter width sweep

`budget_sweep.py` tests whether allocating more of a fixed parameter budget to
embedding width solves queried copying. Every candidate uses the experimental
parameter-free normalization and GELU model, random trainable embeddings,
context length 16, and the same 96-symbol input/output vocabulary as the
previous capability probes. All parameters are counted, including unused
vocabulary rows and biases.

For embedding width d and feed-forward hidden width h, total parameters are

`P = (16 + 96)*d + (d + 96 + 1)*h + 96`.

For each d in 16, 24, 32, 40, 48, 56, 64, 72, and 80, h is the largest positive
integer satisfying P ≤ 10,000. The budget is strict; integer layer sizes leave
some unused capacity. At this vocabulary and context size, a width greater than
86 cannot fit even one hidden unit under the budget. This is a parameter-count
limit for this implementation, not a mathematical bound on required dimensions.

A one-block transformer with embedding width 16 and feed-forward width 162
provides a fresh 9,970-parameter reference. It also learns from random
initialization. All models receive identical generated training batches within
a seed, Adam at 0.001, batch size 64, and 10,000 updates. Each configuration has
three seeds. Scores use the same 4,096 fixed evaluation samples from the
readout experiments (`700000 + context`); no checkpoint selection is performed.
These samples have already been inspected in previous experiments, so this
remains an exploratory comparison. The final checkpoint is always reported.

Run serially:

```sh
python budget_sweep.py --output budget-results.json
```

The actual run uses independent per-seed processes:
`budget-seed-{0,1,2}.json`, with corresponding logs and parameter snapshots.
`python summarize_budget.py` merges and validates those files, checks every
parameter budget, and creates `budget-results.json`, `budget-summary.json`,
and `budget-summary.txt`. CPU timing is not used for comparison.

This sweep keeps the vocabulary and model family fixed. Reducing vocabulary
size, sharing embedding parameters, or factorizing the tables would change
the allocation problem and requires a separate comparison for both model families.

### Strict-budget results

Final queried-copy accuracy at context 16; mean ± sample SD across three
seeds, 10,000 updates. All total parameter counts are strictly below 10,000.

| Model | Embedding dimensions | Hidden width | Total parameters | Accuracy (%) |
|---|---:|---:|---:|---:|
| candidate | 16 | 71 | 9,911 | 18.64 ± 0.41 |
| candidate | 24 | 59 | 9,923 | 18.94 ± 0.58 |
| candidate | 32 | 48 | 9,872 | 21.31 ± 3.45 |
| candidate | 40 | 39 | 9,919 | 20.43 ± 2.33 |
| candidate | 48 | 31 | 9,967 | 22.35 ± 3.11 |
| candidate | 56 | 23 | 9,887 | 19.95 ± 1.33 |
| candidate | 64 | 16 | 9,840 | 19.27 ± 0.40 |
| candidate | 72 | 10 | 9,850 | 17.73 ± 0.73 |
| candidate | 80 | 5 | 9,941 | 18.26 ± 0.70 |
| transformer | 16 | 162 | 9,970 | 100.00 ± 0.00 |

The best tested candidate allocation is width 48 / hidden 31, with 22.35 ±
3.11% accuracy. The modal-token shortcut scores 19.19%, and the one-block
transformer reaches 100% for all three seeds. Increasing embedding dimensions
within this fixed budget does not close the queried-copy gap in these runs.
This is a negative result for the tested allocation and training settings, not
a proof of impossibility at 10,000 parameters.

## Extra hidden layer under the same 10,000-parameter budget

`deep_candidate_model.Model` adds one learned affine layer and a second GELU
activation to the modified model's readout:

`pool → parameter-free normalization → Linear(d,h) → GELU → Linear(h,h) → GELU → Linear(h,96)`.

It retains the original position/token embeddings and multiplication/sum.
This adds a hidden layer to the feed-forward readout, not an attention block
or a second copy of the complete pooling architecture. Both hidden layers use
the same width h. The full parameter count is

`P = 112*d + (d + 97)*h + 96 + h*h + h`.

For each of the previous nine embedding widths (16 through 80), h is the largest
integer keeping P ≤ 10,000. Thus the added depth is paid for by narrower hidden
layers. No normalization parameters, additional preprocessing parameters, or
unreported trainable tensors are excluded from the count.

Each configuration uses the same three seeds, generated training batches,
4,096 evaluation examples, Adam 0.001, batch size 64, and 10,000 updates as the
one-hidden-layer strict-budget sweep. Final checkpoints are reported. The
one-hidden-layer results and validated 9,970-parameter one-block transformer
reference are reused from `budget-results.json`, rather than rerunning
unchanged controls. This comparison tests equal-width two-hidden-layer
readouts; it is not an exhaustive search over asymmetric layer widths or
optimization settings.

Reproduce serially:

```sh
python budget_sweep.py --hidden-layers 2 --skip-reference --output depth-budget-results.json
```

Actual runs use `depth-budget-seed-{0,1,2}.json`, logs and corresponding weight
snapshot directories. `python summarize_depth_budget.py` validates and merges
those files, checks compatibility with the saved shallow controls, and produces
`depth-budget-results.json`, `depth-budget-summary.json`, and
`depth-budget-summary.txt`. All three seeds run independently in parallel;
timing is not compared.

### Added-layer results

Final accuracy (%), mean ± sample SD across three seeds; both hidden layers
in the added-layer model have the listed equal width.

| Dimensions | Two hidden widths | Deep total parameters | One-hidden-layer accuracy | Two-hidden-layer accuracy |
|---|---:|---:|---:|---:|
| 16 | 49 + 49 | 9,875 | 18.64 ± 0.41 | 18.47 ± 0.08 |
| 24 | 43 + 43 | 9,879 | 18.94 ± 0.58 | 19.67 ± 1.67 |
| 32 | 37 + 37 | 9,859 | 21.31 ± 3.45 | 19.22 ± 0.07 |
| 40 | 31 + 31 | 9,815 | 20.43 ± 2.33 | 20.10 ± 2.35 |
| 48 | 26 + 26 | 9,944 | 22.35 ± 3.11 | 24.43 ± 1.02 |
| 56 | 20 + 20 | 9,848 | 19.95 ± 1.33 | 23.01 ± 0.71 |
| 64 | 15 + 15 | 9,919 | 19.27 ± 0.40 | 26.72 ± 8.40 |
| 72 | 10 + 10 | 9,960 | 17.73 ± 0.73 | 31.08 ± 14.92 |
| 80 | 5 + 5 | 9,971 | 18.26 ± 0.70 | 46.31 ± 18.21 |

The best tested deep allocation is d=80, h=5 in each hidden layer, with
9,971 parameters and 46.31 ± 18.21% accuracy. Its individual seeds score
25.29%, 56.23%, and 57.40%. The best tested shallow allocation scores
22.35 ± 3.11% with 9,967 parameters; the saved one-block transformer reference
scores 100% in all three runs with 9,970 parameters. Added depth improves the
wide candidates in this sweep, but substantial variation remains and the
transformer gap is not closed. These are best-of-grid exploratory comparisons,
not significance claims or a comprehensive search over all 10k-parameter networks.

## Sigmoid query gating with one hidden stage

`gated_candidate_model.Model` keeps the normalized-GELU feature branch and adds
a parallel query gate. For per-example input `[positions, tokens]`:

- `z = normalize(sum_i position_embedding[i] * token_embedding[tokens[i]])`
- `q = token_embedding[tokens[-1]]`
- `h = GELU(Wz*z + bz) * sigmoid(Wq*q + bq)`
- `logits = Wo*h + bo`

The last token already encodes the requested position in queried copying.
The gate reuses the existing token embedding table; there is no separate query
embedding table and no answer information. The query branch is not normalized.
Only sigmoid gates are tested, per the user's clarification; GELU remains the
feature activation. Gating is over pooled hidden features, not individual
positions, and is not attention. The two input-to-hidden projections are
parallel; there is one hidden stage, not a second serial hidden layer.

Total parameters for width d and hidden width h are

`P = 112*d + (2*d + 98)*h + 96`.

This includes the gate's weights/biases, both embedding tables, feature
projection and output projection. For each previous embedding width from 16
through 80, h is the largest positive integer keeping P ≤ 10,000. All
initializations and embeddings are learned from scratch.

`gated_budget_sweep.py` uses exactly the same task, context 16, vocabulary 96,
three seeds, training RNG, 4,096 evaluation examples, Adam 0.001, batch size 64,
and 10,000 updates as the prior budget sweeps. Previously validated shallow,
deep and one-block transformer controls are reused. Scores use the final
checkpoint, not the best monitored score. This remains an exploratory study
on previously inspected evaluation data.

Reproduce serially:

```sh
python gated_budget_sweep.py --output gated-budget-results.json
```

Actual runs use `gated-budget-seed-{0,1,2}.json`, separate logs and parameter
snapshots. `python summarize_gated_budget.py` checks compatibility with both
previous controls, validates all parameter counts and completed runs, and
produces `gated-budget-results.json`, `gated-budget-summary.json`, and
`gated-budget-summary.txt`. Independent seeds run concurrently; no new speed
comparison is made. Runtime checks also verify batch independence and output
shape against individually evaluated examples.

### Sigmoid-gate results

Final accuracy (%), mean ± sample SD across three seeds; all gated models
have one hidden stage and at most 10,000 total parameters.

| Dimensions | Hidden units | Parameters | Sigmoid query gate | Ungated one hidden | Ungated two hidden |
|---|---:|---:|---:|---:|---:|
| 16 | 62 | 9,948 | 19.70 ± 0.91 | 18.64 | 18.47 |
| 24 | 49 | 9,938 | 20.43 ± 1.66 | 18.94 | 19.67 |
| 32 | 39 | 9,998 | 23.82 ± 2.19 | 21.31 | 19.22 |
| 40 | 30 | 9,916 | 23.81 ± 2.09 | 20.43 | 20.10 |
| 48 | 23 | 9,934 | 21.26 ± 2.68 | 22.35 | 24.43 |
| 56 | 17 | 9,938 | 19.20 ± 0.36 | 19.95 | 23.01 |
| 64 | 12 | 9,976 | 18.58 ± 0.20 | 19.27 | 26.72 |
| 72 | 7 | 9,854 | 16.37 ± 0.40 | 17.73 | 31.08 |
| 80 | 3 | 9,830 | 26.16 ± 1.85 | 18.26 | 46.31 |

The best tested gated allocation is width 80 / hidden 3, with 9,830 parameters
and 26.16 ± 1.85% accuracy (individual seeds 24.02%, 27.20%, 27.27%). The best
previous ungated shallow candidate averages 22.35%, the best ungated deep
candidate 46.31%, and the one-block transformer reference 100%. The gate gives
a modest best-of-grid gain over the shallow baseline, but does not solve queried
copying or outperform the deeper candidate at this budget. No GELU-gate variant
was tested. These results concern the pooled-feature gate as implemented; they
do not test query-dependent weighting of positions before pooling.

## Larger hidden-depth sweep at fixed parameters and updates

`stacked_candidate_model.Model` generalizes the ungated normalized-GELU readout
to L equal-width hidden layers. It preserves the original learned position and
token embeddings, multiplication and summation, applies parameter-free
normalization once after pooling, then uses GELU after every hidden affine
layer. There are no query gates, residual connections, or additional intermediate
normalizations in this experiment. Output logits remain linear.

`P = 112*d + (d + 97)*h + 96 + (L - 1)*(h*h + h)`.

The new depths are L=3, 4, and 6; the previous L=1 and L=2 sweeps provide saved
controls. Each depth uses the same nine embedding dimensions (16, 24, 32, 40,
48, 56, 64, 72, 80), choosing the largest equal hidden width h such that total
P ≤ 10,000. Runtime assertions count actual model parameters and compare them
with the formula. A separate check confirms the generalized model at depth two
reproduces the existing two-hidden-layer implementation at identical initialization.

All runs keep context 16, vocabulary 96, queried copying, Adam 0.001, batch 64,
10,000 updates, and the same three training/evaluation seeds. Final checkpoints
are reported. The same evaluation data have been used in earlier exploratory
experiments; no new blind-test claim is made. This tests the depth/width allocation
under the existing optimizer, not all possible deep architectures or schedules.
The existing one-block 9,970-parameter transformer result is reused unchanged.

Example serial commands:

```sh
python budget_sweep.py --hidden-layers 3 --skip-reference --output stack-depth-3.json
python budget_sweep.py --hidden-layers 4 --skip-reference --output stack-depth-4.json
python budget_sweep.py --hidden-layers 6 --skip-reference --output stack-depth-6.json
```

Actual runs partition by seed, with depths 3, 4, and 6 sequential within each
seed job, and the three seed jobs concurrent. Raw files are
`stack-depth-{3,4,6}-seed-{0,1,2}.json`, with separate logs and weight directories.
`summarize_layer_depth.py` validates 81 new runs, reuses 57 saved control runs,
and writes `layer-depth-results.json`, `layer-depth-summary.json`, and
`layer-depth-summary.txt`. Both the full grid and the best observed mean per
depth are retained. Best-per-depth values are selected over nine allocations;
they are descriptive exploratory results, not significance claims. Timing is
not compared across concurrent runs.

After the depth grid is complete, `python validate_layer_depth.py` selects the
single candidate configuration with the highest earlier three-seed mean and
reloads its final weights. It first reproduces the recorded original-sample
metrics, then evaluates that fixed configuration and the saved transformer
reference on 16,384 newly generated examples (RNG seed 20261121). There is no
additional training or tuning after this fresh sample is generated. Results
are written to `layer-depth-fresh-validation.json`. This provides a fresh-input
check after the earlier adaptive grid comparisons; the same three training
seeds are reused, so it does not test new random initializations.

### Full depth-sweep results

Best observed mean at each depth, selected from the same nine embedding widths.
Each configuration uses three seeds, 10,000 updates, and ≤10,000 total parameters.

| Hidden layers | Embedding dimensions | Hidden width per layer | Parameters | Accuracy (%) |
|---|---:|---:|---:|---:|
| 1 | 48 | 31 | 9,967 | 22.35 ± 3.11 |
| 2 | 80 | 5 | 9,971 | 46.31 ± 18.21 |
| 3 | 80 | 4 | 9,804 | 55.49 ± 19.22 |
| 4 | 72 | 9 | 9,951 | 83.33 ± 12.45 |
| 6 | 64 | 12 | 9,976 | 98.01 ± 0.95 |
| Transformer, one block | 16 | 162 (feed-forward) | 9,970 | 100.00 ± 0.00 |

Full grid, mean accuracy (%):

| Dimensions | 1 hidden | 2 hidden | 3 hidden | 4 hidden | 6 hidden |
|---|---:|---:|---:|---:|---:|
| 16 | 18.64 | 18.47 | 18.81 | 20.86 | 23.64 |
| 24 | 18.94 | 19.67 | 19.17 | 22.92 | 26.88 |
| 32 | 21.31 | 19.22 | 22.75 | 24.81 | 48.17 |
| 40 | 20.43 | 20.10 | 24.43 | 29.70 | 55.22 |
| 48 | 22.35 | 24.43 | 23.14 | 56.95 | 82.54 |
| 56 | 19.95 | 23.01 | 30.08 | 41.26 | 89.78 |
| 64 | 19.27 | 26.72 | 36.13 | 74.32 | 98.01 |
| 72 | 17.73 | 31.08 | 53.51 | 83.33 | 94.73 |
| 80 | 18.26 | 46.31 | 55.49 | 64.96 | 90.91 |

The strongest tested allocation has six hidden layers, width 64 embeddings,
12 units per hidden layer, and 9,976 parameters. The original-sample accuracy
is 98.01 ± 0.95%, with individual seeds 97.56%, 97.36%, and 99.10%. The
transformer uses 9,970 parameters. This closes much of the earlier gap on this
task, but does not exceed the transformer, and additional depth does not
improve every fixed-width configuration monotonically. No claim about language
prediction is made for this six-layer model; its language benchmark remains untested.

Fresh-input validation after selecting the above configuration:

| Model | Fresh accuracy (%) | Individual seeds (%) |
|---|---:|---|
| selected_candidate | 98.15 ± 0.75 | 97.79, 97.65, 99.01 |
| transformer | 100.00 ± 0.00 | 100.00, 100.00, 100.00 |

The fresh check uses 16,384 examples generated after architecture selection,
with no further training or tuning. All three selected-model checkpoints also
reproduce their recorded original-sample metrics after loading. All 81 new
fits and the reused controls pass parameter-count and finite-metric checks.

## Isolated speed measurement of the selected six-layer model

`speed_benchmark.py` compares the trained six-hidden-layer model (64 embedding
dimensions, 12 units per hidden layer, 9,976 parameters) with the trained
one-block transformer (16 dimensions, feed-forward width 162, 9,970 parameters).
Both use their seed-0 final checkpoint. The readout has no residual connections;
the transformer retains its original pre-norm and residual design. Architectures
are not changed for timing.

Hardware: Intel Core Ultra 7 155H, WSL2 Linux, JAX CPU backend, FP32. The two
models run sequentially in one process, with affinity 0–21 (all 22 logical CPUs
available), OMP_NUM_THREADS=22 and OPENBLAS_NUM_THREADS=1. There are no concurrent
experiment jobs. This makes the CPU settings identical; these tiny models need
not saturate all cores. Full software/platform details are in `speed-results.json`.

Models and 32 pre-generated input batches are device-resident before timing.
Context length is 16, vocabulary is 96. Seven timed trials alternate AB/BA model
order. Reported times are medians over trials; ranges, SDs and all trials are
retained. Data generation, input transfers, checkpoint loading, initialization,
tracing and compilation are excluded from steady-state times.

Inference: batch sizes 1 and 64, 20 warmup calls, then 500 calls per timed trial.
Each request blocks until its output is ready. This measures request latency
including NNX/JIT dispatch, rather than pure arithmetic-kernel time.

Python-driven training: batch 64, Adam 0.001, 200 updates per timed trial. Each
trial clones the same checkpoint and initializes fresh optimizer state, then
warms up for 20 updates. The final loss and all model/optimizer state arrays
are synchronized before stopping the clock. Data sampling is excluded, and
these timing-only updates are discarded.

Compiled training: the same model, loss and Adam updates run in a JAX `lax.scan`
block of 200 updates, with dynamic input arrays and model/optimizer state carried
through all updates. One warmup block precedes the seven timed blocks. This
removes per-update Python dispatch. Five updates are checked against the usual
training function: losses and every model/Adam-state leaf agree within
rtol=1e-4, atol=1e-5. Timing includes full state/output synchronization.
This is a different execution mode from Python-driven training; comparisons
between models are made within each mode.

| Measurement | Six-layer model (ms) | Transformer (ms) | Speedup |
|---|---:|---:|---:|
| Inference, batch 1 | 0.762 | 0.773 | 1.01× (essentially equal) |
| Inference, batch 64 | 0.761 | 1.171 | 1.54× |
| Python-driven training update, batch 64 | 1.961 | 2.446 | 1.25× |
| Compiled-loop training update, batch 64 | 0.352 | 1.994 | 5.67× |

Compilation/first-call times are stored separately. For inference and the
Python training function, these entries include the first execution as well as
tracing/compilation. For compiled training, explicit `lower(...).compile()` is
timed without execution (about 0.373 s for the candidate and 0.590 s for the
transformer). The one-trial implementation check is preserved as
`speed-smoke.json`; final figures use only the seven-trial `speed-results.json`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=22 taskset -c 0-21 python speed_benchmark.py > speed-benchmark.log 2>&1
python summarize_speed.py
```

`speed-summary.txt` is the formatted result. These measurements apply to this
hardware, context, batch sizes, backend and implementation. Faster updates do
not establish faster time to target accuracy: the transformer learned queried
copying in fewer updates, and its fresh-sample accuracy is 100%, versus 98.15%
for the selected readout. This is not a long-context, GPU, cached autoregressive
generation, or language-model serving benchmark.

## Paired residual-connection experiment

`stacked_candidate_model.Model(..., hidden_layers=6, residual=True)` uses the
same six-hidden-layer, 64-dimensional, 12-unit architecture with parameter-free
skip connections at the five equal-width hidden transitions:

`h_next = h + GELU(W*h + b)`.

There is no skip over the first 64→12 projection or the final 12→96 projection.
The experiment does not add intermediate normalization, scaling, gates, or
zero-initialize branches. The only change is the five additions. All initial
learned weights match exactly between each paired plain/residual model; both
contain 9,976 parameters. This tests simple residual addition, not every residual
block design or a prenorm residual architecture.

`residual_experiment.py` reruns the non-residual control alongside residuals,
using three seeds, identical batches, Adam 0.001, batch 64, and 10,000 updates.
For both models, the entire training loop runs as compiled JAX `lax.scan`, with
state carried across chunks ending at updates 1,000, 3,000, and 10,000. All
batches are pregenerated using the previous training RNG and placed in memory;
there is no per-update Python sampling or dispatch. The optimizer is not reset
between chunks. Five ordinary and compiled updates agree in their losses and
every model/Adam-state leaf within rtol=1e-4, atol=1e-5. The new plain controls
also reproduce all three original-sample accuracy values from the prior sweep.

The new validation uses 16,384 IID examples at seed 20261122. Both candidate
variants and all three saved transformer checkpoints are evaluated on these
same inputs, without additional tuning or training.

| Model | Parameters | Original evaluation accuracy (%) | New validation accuracy (%) |
|---|---:|---:|---:|
| Plain six-layer | 9,976 | 98.01 ± 0.95 | 98.08 ± 0.76 |
| Simple residual six-layer | 9,976 | 96.12 ± 3.18 | 96.06 ± 3.45 |
| One-block transformer | 9,970 | saved reference: 100% | 100.00 ± 0.00 |

Means ± sample SD across three seeds. Residual validation accuracies are
98.68%, 97.34%, and 92.15%; the corresponding plain values are 97.82%, 97.49%,
and 98.94%. The residual form did not improve average accuracy or consistency
under this initialization, architecture and optimizer. This is not a conclusion
about all residual designs or hyperparameters.

The user confirmed inclusion of the timing run that had already completed.
The report includes those saved timings; no new timing run was performed.

```sh
python residual_experiment.py --seed 0 --output residual-seed-0.json
python residual_experiment.py --seed 1 --output residual-seed-1.json
python residual_experiment.py --seed 2 --output residual-seed-2.json
python summarize_residual.py
```

Actual training seed jobs ran concurrently. `residual-results.json` holds the
paired results; `residual-summary.json` and `residual-summary.txt` include accuracy and the saved timings.
Raw seed files, parameter snapshots and the five-update implementation check
(`residual-smoke.json`) are retained. The original model implementation is retained.

### Saved residual timing results

The three models ran sequentially in one isolated process after training had
finished, using all 22 logical CPUs, seven warmed-up trials, and rotating/
reversing model order. Compilation, data generation and input transfers were
excluded. These are seed-0 checkpoint timings, not three-seed timing averages.

| Median milliseconds | Plain | Residual | Transformer |
|---|---:|---:|---:|
| Inference, batch 1 | 0.742 | 0.768 | 0.764 |
| Inference, batch 64 | 0.733 | 0.759 | 1.170 |
| Python-driven training update, batch 64 | 1.870 | 1.940 | 2.319 |
| Compiled-loop training update, batch 64 | 0.334 | 0.337 | 1.851 |

The simple skip connections have little speed overhead, but provided no
accuracy improvement in this paired test. The non-residual model remains the
stronger tested choice here. This result does not evaluate prenorm residual
blocks, residual scaling or alternate initialization. The raw saved timings
are in `residual-speed-results.json`; `summarize_residual.py` reads those data
without performing any new speed measurement.

## Retained best-copy architecture tested on language

`best_model.py` and `best-model.json` preserve the selected six-hidden-layer,
64-dimensional, 12-unit normalized-GELU architecture without residuals or query
gating. Its three synthetic-task checkpoints remain available. New language
weights are trained from scratch because the context/vocabulary sizes and token
meaning differ; the synthetic checkpoints are not transferred.

The language test uses Tiny Shakespeare, the same data checksum and contiguous
80/10/10 split as before, context length 32, 65 character types, Adam 0.001,
batch 64, and 10,000 updates. Three seeds share identical training batches and
evaluation windows across models. One next-character label per context is
supervised for every architecture. Final checkpoints are used, without choosing
a checkpoint by validation performance.

The exact retained architecture now contains 8,613 parameters because its
embedding tables and output classes change with the task. An exactly matched
shallow control uses the same 64-dimensional embeddings and one 18-unit hidden
layer, also totaling 8,613 parameters. The one-block transformer uses width 16,
four heads and feed-forward width 144, totaling 8,609 parameters. These are
smaller than the earlier approximately 18k-parameter language comparison, and
the earlier perplexity numbers should not be treated as budget-matched here.

All three models use the same compiled `lax.scan` training implementation.
Five-step loss and complete model/Adam-state parity checks against ordinary
updates pass for every run. Validation curves are recorded at updates 1,000,
3,000 and 10,000; the test split is scored only after the final update. Evaluation
uses the original fixed 4,096 windows per split and may contain overlapping
windows. This is a small character-level comparison, not a large-scale language
benchmark or a blind test after all earlier exploratory work.

| Model | Parameters | Test perplexity | Test accuracy (%) |
|---|---:|---:|---:|
| deep | 8,613 | 9.78 ± 0.04 | 35.31 ± 0.34 |
| shallow | 8,613 | 8.94 ± 0.26 | 37.65 ± 0.65 |
| transformer | 8,609 | 8.43 ± 0.16 | 38.25 ± 0.56 |

Means ± sample SD across three seeds. The architecture that was strongest
on queried copying is not the strongest of these language models at this
allocation. The matched shallow model and transformer both perform better on
next-character prediction. The result does not establish that every deeper
language architecture is inferior; dimensions and hidden width were retained
from the copy-task selection, rather than retuned for language.

`language_depth_test.py` runs the comparison. Raw seed outputs, logs and
checkpoints use `language-depth-seed-{0,1,2}`. `summarize_language_depth.py` checks
all nine runs, parameter counts, dataset/split consistency and finite metrics,
then writes `language-depth-results.json`, `language-depth-summary.json`, and
`language-depth-summary.txt`. The text report includes autoregressive examples
at temperature 0.8 using the same prompt and random seed for each model.
`language-vocabulary.json` preserves the ordered character mapping. `best-model.json`
points to the new language checkpoint files separately from the copy checkpoints.

```sh
python language_depth_test.py --seed 0 --output language-depth-seed-0.json
python language_depth_test.py --seed 1 --output language-depth-seed-1.json
python language_depth_test.py --seed 2 --output language-depth-seed-2.json
python summarize_language_depth.py
```

Actual seeds ran concurrently. Training execution-time fields are diagnostic
and are not used for a speed comparison. No isolated timing benchmark was run
for this language experiment. The five-update smoke check is preserved as
`language-depth-smoke.json` and is excluded from final results.

## Language-selected 16-dimensional model on queried copying

The language-selected architecture (16 embedding dimensions, six hidden layers
of 30 units, normalization and GELU, no residuals) was trained from scratch on
queried copying for 10,000 compiled Adam updates across three seeds. It has
9,928 parameters with the copy task's context 16 and vocabulary 96. It uses the
same training RNG, 4,096-example evaluation sample and 16,384-example validation
sample as the prior matched comparisons. The original evaluation accuracy is
23.64 ± 1.57%; the 16,384-example validation accuracy is 24.07 ± 0.81%. The
64-dimensional, 12-unit-per-layer model reached 98.08 ± 0.76% on that same larger
sample; the one-block transformer reached 100%. The language-selected model did
not retain its language advantage on this retrieval task.

`query_language_config.py` implements the test; `query-language-results.json`
contains all three runs, and individual checkpoint files are in the corresponding
`query-language-seed-{0,1,2}` directories.

## Independent rerun of the 16-dimensional queried-copy test

The latest 16-dimension, six-hidden-layer, 30-unit, 9,928-parameter test was
reviewed and rerun unchanged for seeds 0, 1 and 2 with 10,000 compiled updates.
All recorded checkpoint metrics and final validation metrics reproduce the
previous run exactly. Validation accuracy is 24.07 ± 0.81%, with individual
values 23.27%, 24.90%, and 24.05%. The rerun is saved separately in
`sol-query-language-results.json`; earlier results and checkpoints are preserved.

## Residual six-layer language test

Language residual test: context 32, vocabulary 65; three seeds, 10,000 compiled Adam updates.
Six hidden layers of 12 units, embedding width 64; five hidden-to-hidden residuals.
No residual on input or output projections; no intermediate normalization or scaling.

six-layer plain: params=8613, perplexity=9.78 +/- 0.04, accuracy=35.31% +/- 0.34
six-layer residual: params=8613, perplexity=9.43 +/- 0.16, accuracy=36.39% +/- 0.25
one-layer plain: params=8613, perplexity=8.94 +/- 0.26, accuracy=37.65% +/- 0.65
transformer: params=8609, perplexity=8.43 +/- 0.16, accuracy=38.25% +/- 0.56

The residual variant improves over the same deep model without skips on language,
but remains behind the matched shallow model and transformer. This differs from
the queried-copy residual result. All models use the same data splits, batches,
optimizer and final-update evaluation protocol. No speed benchmark was run.

## Parameter-free pre-norm residual language test

Language test: six hidden layers of 12 units, embedding width 64, three seeds, 10,000 compiled Adam updates.
Pre-norm residual: h + GELU(W*normalize(h)+b); parameter-free normalization, epsilon 1e-5.

Six-layer plain: 8613 params, perplexity 9.78 +/- 0.04, accuracy 35.31% +/- 0.34
Simple residual: 8613 params, perplexity 9.43 +/- 0.16, accuracy 36.39% +/- 0.25
Pre-norm residual: 8613 params, perplexity 9.67 +/- 0.25, accuracy 35.76% +/- 0.50
Transformer: 8609 params, perplexity 8.43 +/- 0.16, accuracy 38.25% +/- 0.56

All five hidden residual branches normalize their input before the affine layer.
Initial input/output projections do not have skips. Dataset, batches, optimizer,
seeds and final-update evaluation match the earlier language tests. Five-step
compiled/ordinary parity checks pass for all three runs. This modification did
not improve over simple residuals in this tested narrow architecture. No speed
benchmark was run.

## Shared 64-dimensional architecture across tasks

Same architecture per task: 64 dims, normalized GELU, no residuals; 10,000 updates, three seeds.
Vocabulary/context change by task; layer widths stay identical.

{'depth': 8, 'dimensions': 64, 'hidden': 11, 'copy_params': 9959, 'language_params': 8627, 'copy_accuracy': 97.344970703125, 'copy_sd': 3.493914857567773, 'language_perplexity': 10.600421712660262, 'language_sd': 0.352003507823456}
{'depth': 12, 'dimensions': 64, 'hidden': 9, 'copy_params': 9703, 'language_params': 8433, 'copy_accuracy': 98.02042643229167, 'copy_sd': 3.3230284260498184, 'language_perplexity': 11.51596860567463, 'language_sd': 0.08384692868127229}

Eight layers use 11 units each; twelve use 9 units each. Both tasks retain
these exact layer shapes and share the usual data/optimizer/evaluation protocols.
Only vocabulary and context differ. Initial 16/32-dimension jobs were cancelled
after the user clarified 64 dimensions; their partial artifacts are excluded.

## Single- and multi-query associative recall

Five stored key/value pairs; single query versus three sequential queries.
Ours: six hidden layers, 64 embedding dims, 12 units each, no residuals, 9,976 parameters.
Transformer: one block, 16 dims, four heads, FF width 162, 9,970 parameters.
Three seeds; 10,000 compiled Adam updates, batch 64, learning rate .001.
Single query supervises one value/context; multi query supervises three values/context.
Teacher-forced causal prefixes: no current or future answer is visible.
Evaluation: same new 16,384 sequences for both models; mean +/- sample SD.
This is an MQAR-style variant, not the official benchmark.

single / ours: accuracy 23.42% +/- 2.13; all answers correct 23.42%
single / transformer: accuracy 30.02% +/- 0.11; all answers correct 30.02%
single / frequency shortcut: accuracy 30.57% +/- 0.00; all answers correct 30.57%
multi / ours: accuracy 29.73% +/- 1.09; all answers correct 1.74%
multi / transformer: accuracy 31.49% +/- 1.54; all answers correct 1.05%
multi / frequency shortcut: accuracy 30.68% +/- 0.00; all answers correct 0.35%

Neither model solves these settings: query accuracy is near the frequency
shortcut. This does not establish architectural equivalence or a general
transformer limitation. The comparison uses a single attention block; deeper
attention configurations have not been tested on this exact task. Raw outputs
are in `associative-single-seed-*` and `mqar-seed-*`; `mqar_test.py` generates and
trains the tasks. The paper remains paused.

## Six-hidden-layer transformer FFN comparison

Six feed-forward hidden layers in both models; one attention block in transformer.
Ours: embedding 64 / hidden 12. Transformer: embedding 16 / hidden 28.
Queries: ours 9,976 params; transformer 9,608. Language: ours 8,613; transformer 8,841.
Three seeds, 10,000 Adam updates, batch 64, identical data per task.

Task/metric         Ours              Transformer 1 hidden  Transformer 6 hidden
position/accuracy   98.08 +/- 0.76    100.00 +/- 0.00       99.99 +/- 0.01
single/accuracy     23.42 +/- 2.13    30.02 +/- 0.11        30.11 +/- 0.23
multi/accuracy      29.73 +/- 1.09    31.49 +/- 1.54        32.10 +/- 0.25
language/perplexity 9.78 +/- 0.04     8.43 +/- 0.16         8.73 +/- 0.11

The transformer retains one causal attention block, prenorm and block residuals.
Its FFN has an input projection, five equal-width hidden projections, and an
output projection, with GELU after each hidden layer. All seven affine maps
are counted. No intermediate FFN normalizations or skips are added. Both
architectures use the same internal shapes across tasks. Total parameters are
close rather than exact (difference below 4%). Earlier one-hidden-layer
transformer and our-model controls are reused. This is not an official MQAR
benchmark, a multi-attention-block test, or a speed comparison.

### Execution optimization for the 100k transformer

`benchmark_100k.py` uses an output-position optimization for the requested
one-block transformer. It retains all input embeddings, keys and values, but
computes attention outputs and feed-forward activations only at supervised
positions. The causal mask excludes current/future answers in associative tasks.
For multiple queries, the positions are handled together rather than repeating
full-prefix forward passes. Architecture, weights and loss are unchanged.
`check_fast_transformer.py` verifies logits (rtol=1e-4, atol=1e-5) and parameter
gradients (rtol=1e-3, atol=1e-5) against the original path for all four tasks.
The transformer was restarted from its original random initialization using
this implementation; partial original logs/results were archived separately.
Our completed runs use the same compiled optimizer loop as before.

Earlier speed figures describe the earlier implementation. They should not be
presented as an optimized-transformer performance comparison without rerunning
those timings against this output-position optimization. No such speed claim
is made by the 100k quality experiment. The agreed paper title is preserved in
`paper/title.txt`; the paper remains paused pending these experiments.

## Approximately 100k parameters, one-block transformer

Shared architecture across tasks, approximately 100k parameters.
Ours: 64 dims, six hidden layers of 120 units, normalization + GELU, no residuals.
Transformer: 64 dims, one block, four heads, standard one-hidden-layer FFN of 536 units.
Three seeds, 10,000 compiled Adam updates, batch 64, learning rate .001.
Same data and evaluation samples as previous tests. No speed benchmark.
Parameter counts change with context/vocabulary; internal shapes stay fixed.

position / ours: params=99184, accuracy=97.61% +/- 1.38
position / transformer: params=99640, accuracy=100.00% +/- 0.00
single / ours: params=99184, accuracy=29.07% +/- 0.80
single / transformer: params=99640, accuracy=29.25% +/- 0.58
multi / ours: params=99184, accuracy=34.36% +/- 0.33
multi / transformer: params=99640, accuracy=30.66% +/- 0.12
language / ours: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / transformer: params=96665, accuracy=43.47% +/- 0.36, perplexity=7.31 +/- 0.12

Language uses the previous Tiny Shakespeare split and context 32. Position copying
uses context 16. Single and multiple associative queries use five stored pairs
and causal prefixes, as in `mqar_test.py`. Single queries supervise one label
per context; multi queries supervise three. The paper remains paused. This
retains the one-block standard-FFN transformer requested by the user and does
not claim an exhaustive transformer architecture search. Final checkpoints
are scored, with no best-step selection. Runs and weights are in `large-*`;
`benchmark_100k.py` and `summarize_100k.py` reproduce the experiment.

### 100k quality results

Means ± sample SD across three seeds, all after 10,000 compiled updates.

| Test | Our model | One-block standard-FFN transformer |
|---|---:|---:|
| position accuracy (%) | 97.61 ± 1.38 | 100.00 ± 0.00 |
| single accuracy (%) | 29.07 ± 0.80 | 29.25 ± 0.58 |
| multi accuracy (%) | 34.36 ± 0.33 | 30.66 ± 0.12 |
| Language perplexity | 7.50 ± 0.15 | 7.31 ± 0.12 |
| language accuracy (%) | 43.25 ± 0.38 | 43.47 ± 0.36 |

The language accuracy gap is about 0.22 percentage points, with perplexity
7.50 versus 7.31. This is a much smaller gap than at the smaller selected
configuration, but does not establish equivalence or superiority. Associative
retrieval remains weak: the modal-value shortcut scores about 30.6%. Our
multi-query score exceeds that shortcut modestly, while neither model solves
retrieval reliably. Position copying does not improve over the smaller model.
No residual variant or isolated speed benchmark was run at this scale.

## Fixed sinusoidal positions, approximately 10k parameters

Fixed sinusoidal positions, <=10k trainable parameters, three seeds.
Mean +/- sample standard deviation; 10,000 Adam updates, batch64, lr .001.
Ours: d64, six hidden layers, h15 queries / h23 language; no residuals.
Transformer: d16, four heads, one standard block, FF170 queries / FF201 language.
Same data/evaluation as previous tests. Hidden widths use the freed positional budget.
This is a matched-budget comparison, not an isolated positional-encoding ablation.
No speed measurement or long-context evaluation.

position / ours: 9855 params, accuracy 21.11% +/- 0.54
position / transformer: 9978 params, accuracy 100.00% +/- 0.00
single / ours: 9855 params, accuracy 26.99% +/- 1.26
single / transformer: 9978 params, accuracy 29.77% +/- 0.19
multi / ours: 9855 params, accuracy 30.30% +/- 0.64
multi / transformer: 9978 params, accuracy 31.15% +/- 0.60
language / ours: 9975 params, accuracy 27.19% +/- 0.49, perplexity 13.10 +/- 0.15
language / transformer: 9978 params, accuracy 37.70% +/- 0.84, perplexity 8.60 +/- 0.21

Reproduce with `sinusoidal_test.py --architecture ours|transformer --seed 0|1|2`, then `summarize_sinusoidal.py`. Logit/gradient parity for the optimized Transformer and complete five-step compiled/ordinary optimizer-state parity for both models pass in `check_sinusoidal.py`. Position weights multiply token embeddings in our model and add to token embeddings in the Transformer. Learned embeddings are replaced, not retained. All trainable weights are counted; fixed sinusoids contribute zero parameters. No paper changes were made.

## Final learned LayerNorm ablation

Paired final LayerNorm ablation: learned positions, d64, h12, six hidden layers.
Three seeds, 10,000 compiled Adam updates, batch64, lr .001.
Only change: learned LayerNorm (epsilon1e-6) before final output linear; +24 parameters.
Mean +/- sample standard deviation. No speed benchmark.

position / plain: 9976 params, accuracy 98.08% +/- 0.76
position / final: 10000 params, accuracy 86.50% +/- 3.29
single / plain: 9976 params, accuracy 23.42% +/- 2.13
single / final: 10000 params, accuracy 27.82% +/- 0.43
multi / plain: 9976 params, accuracy 29.73% +/- 1.09
multi / final: 10000 params, accuracy 33.14% +/- 0.39
language / plain: 8613 params, accuracy 35.31% +/- 0.34, perplexity 9.78 +/- 0.04
language / final: 8637 params, accuracy 36.58% +/- 0.87, perplexity 9.41 +/- 0.15

Reproduce using `final_norm_test.py --architecture plain|final --seed 0|1|2` and `summarize_final_norm.py`. Both baselines were rerun with paired shared initial weights, batches, and evaluation examples. `check_final_norm.py` verifies baseline output parity, identical shared initial parameters, +24 parameters, and five-step compiled versus ordinary loss and complete optimizer/model-state parity for both variants. Data/evaluation protocol matches `benchmark_100k.py`; architecture is the retained small model. No paper changes were made.

## Parameter-free final LayerNorm comparison

Paired final LayerNorm ablation: learned positions, d64, h12, six hidden layers.
Three seeds, 10,000 compiled Adam updates, batch64, lr .001.
Variants: plain=no final norm; final=learned scale/bias (+24 params); fixed=no scale/bias (+0 params). All final norms use epsilon1e-6.
Mean +/- sample standard deviation. No speed benchmark.

position / plain: 9976 params, accuracy 98.08% +/- 0.76
position / final: 10000 params, accuracy 86.50% +/- 3.29
position / fixed: 9976 params, accuracy 82.14% +/- 4.66
single / plain: 9976 params, accuracy 23.42% +/- 2.13
single / final: 10000 params, accuracy 27.82% +/- 0.43
single / fixed: 9976 params, accuracy 27.47% +/- 0.67
multi / plain: 9976 params, accuracy 29.73% +/- 1.09
multi / final: 10000 params, accuracy 33.14% +/- 0.39
multi / fixed: 9976 params, accuracy 33.03% +/- 0.56
language / plain: 8613 params, accuracy 35.31% +/- 0.34, perplexity 9.78 +/- 0.04
language / final: 8637 params, accuracy 36.58% +/- 0.87, perplexity 9.41 +/- 0.15
language / fixed: 8613 params, accuracy 36.48% +/- 0.43, perplexity 9.49 +/- 0.04

Reproduce using `final_norm_test.py --architecture plain|final|fixed --seed 0|1|2` and `summarize_fixed_final_norm.py`. Previous paired plain/learned results are reused; the parameter-free variant uses identical shared initial weights, batches, and evaluation examples. `check_final_norm.py` verifies baseline output parity, identical shared initial parameters, +24 parameters, and five-step compiled versus ordinary loss and complete optimizer/model-state parity for all three variants. Data/evaluation protocol matches `benchmark_100k.py`; architecture is the retained small model. No paper changes were made.

## One versus multiple Transformer blocks, same small budget

Transformer block-count comparison at <=10k total trainable parameters.
d16, four heads, learned positions, one standard FF hidden layer per block.
Three seeds, 10,000 compiled Adam updates, batch64, lr .001.
Same initial seed and training/evaluation examples; FF widths adjusted to budget.
All preceding blocks compute every state; final block computes only supervised outputs.
Mean +/- sample standard deviation. No speed benchmark.

position / 1 blocks: 9970 params, FF 162, accuracy 100.00% +/- 0.00
position / 2 blocks: 9950 params, FF 63, accuracy 100.00% +/- 0.00
position / 4 blocks: 9976 params, FF 14, accuracy 100.00% +/- 0.00
single / 1 blocks: 9970 params, FF 162, accuracy 30.02% +/- 0.11
single / 2 blocks: 9950 params, FF 63, accuracy 29.73% +/- 0.21
single / 4 blocks: 9976 params, FF 14, accuracy 29.45% +/- 0.68
multi / 1 blocks: 9970 params, FF 162, accuracy 31.49% +/- 1.54
multi / 2 blocks: 9950 params, FF 63, accuracy 34.57% +/- 0.05
multi / 4 blocks: 9976 params, FF 14, accuracy 34.37% +/- 0.03
language / 1 blocks: 9995 params, FF 186, accuracy 38.24% +/- 0.52, perplexity 8.36 +/- 0.13
language / 2 blocks: 9975 params, FF 75, accuracy 38.70% +/- 0.50, perplexity 8.43 +/- 0.09
language / 4 blocks: 9869 params, FF 19, accuracy 37.64% +/- 0.67, perplexity 8.62 +/- 0.11

Reproduce with `transformer_blocks_test.py --architecture one|two|four --seed 0|1|2`, then `summarize_transformer_blocks.py`. Every trainable embedding, norm, attention, FFN, and output parameter is counted. Context32 for language;16 for query tasks, using the same generators and evaluation samples as `benchmark_100k.py`. Optimized final-block logits/gradients and five-step full compiled model/optimizer-state parity are checked in `check_transformer_blocks.py`. Fixed parameters imply narrower FFNs at greater depths, so this tests budget allocation rather than adding blocks at constant width throughout. Previous main one-block reference remains unchanged; no paper edits made.

## Two-block Transformer is the default baseline; repeated CPU timing

Approximately 100k parameters: pooled model vs two-block standard Transformer.
Context16, vocab96, FP32 CPU, 22 logical CPUs, seven alternating warmed trials.
Ours: d64 h120 K6, no residual or final norm, 99184 params; Transformer: d64, FF200, 2 blocks, 4 heads, 99056 params.
Learned positions in both; seed0 position-copy checkpoints. Two-block checkpoint trained 10000 updates; quality not a three-seed comparison.
Resident inputs, compilation excluded, synchronized results; no KV cache. First block computes all causal states; final block only supervised outputs.
Five-step complete model/Adam state and losses match ordinary updates; optimized logits/gradients checked against full Transformer.

inference_batch_1: ours 1.570 ms, TF 2.587 ms, speedup 1.65x
inference_batch_64: ours 1.970 ms, TF 4.211 ms, speedup 2.14x
python_training_batch_64: ours 3.357 ms, TF 8.977 ms, speedup 2.67x
compiled_training_batch_64: ours 1.415 ms, TF 7.445 ms, speedup 5.26x

Batch1 inference trial variation: ours 0.800-1.690 ms, TF 1.651-2.660 ms. Absolute timings differ from earlier runs; compare models within this run. These are per-update/request timings, not time to equal quality. No long-context or cache/memory measurements.

Reproduce checkpoint with `two_block_100k.py --architecture transformer --seed 0`, then run `two_block_speed.py` isolated with OMP_NUM_THREADS=22, OPENBLAS_NUM_THREADS=1, CPU affinity0-21. Existing one-block historical results and paper timings remain labeled as such. No paper edits made.

## Learned affine pooled LayerNorm, query-copy and language

Pooled learned-affine LayerNorm comparison, one hidden layer, learned positions.
Query-copy: d255 h240, 113136 base params / 113646 affine params.
Language: d32 h152, 18065 base params / 18129 affine params.
Three seeds, 10000 Adam updates, batch64, lr .001. Epsilon1e-5; final norm unchanged (none).
Query controls reuse original four random-embedding diagnostics. New copy scores use the identical 4096 evaluation examples; extra fresh16384 scores are in raw files.
All six language variants rerun with identical data, held-out windows and shared initialization.
No speed benchmark.

position / sigmoid: params=113136, accuracy=18.59% +/- 0.56
position / norm_sigmoid: params=113136, accuracy=58.27% +/- 1.73
position / affine_norm_sigmoid: params=113646, accuracy=66.89% +/- 1.87
position / gelu: params=113136, accuracy=41.38% +/- 4.35
position / norm_gelu: params=113136, accuracy=98.02% +/- 0.35
position / affine_norm_gelu: params=113646, accuracy=97.81% +/- 0.42
language / sigmoid: params=18065, accuracy=38.12% +/- 0.27, perplexity=8.19 +/- 0.07
language / norm_sigmoid: params=18065, accuracy=38.50% +/- 0.23, perplexity=8.22 +/- 0.08
language / affine_norm_sigmoid: params=18129, accuracy=39.75% +/- 0.14, perplexity=7.81 +/- 0.08
language / gelu: params=18065, accuracy=38.92% +/- 0.34, perplexity=8.17 +/- 0.06
language / norm_gelu: params=18065, accuracy=40.36% +/- 0.25, perplexity=7.81 +/- 0.12
language / affine_norm_gelu: params=18129, accuracy=40.60% +/- 0.32, perplexity=7.85 +/- 0.10

Implementation: `learned_pool_norm_test.py`; aggregation: `summarize_pool_norm.py`. `check_learned_pool_norm.py` verifies identical shared parameters and initial logits between parameter-free and affine normalization, expected extra parameter counts, and five-step full compiled versus ordinary loss/Adam-state parity for sigmoid and GELU. Learned scale/bias use the exact previous normalization expression, then per-coordinate affine transformation, avoiding a changed normalization kernel/epsilon as a confound. Main architecture and paper remain unchanged.

## Deeper pooling at approximately100k parameters

Deeper pooled models, <=100k parameters; d64, learned positions.
Ours: GELU, parameter-free pooled normalization, no residuals/gates/final norm.
Pooled hidden layers/width: 6/120, 8/103, 12/84. Internal dimensions fixed across tasks.
Transformer: 2 standard blocks, d64, 4 heads, FF200 per block across all tasks.
Three seeds, 10000 compiled Adam updates, batch64, lr .001. Same training/evaluation examples.
Six-layer results reused from original100k runs; deeper/2-block reference newly trained. Final checkpoints only.
Context32 language; context16 retrieval. No speed measurement.

position / six: params=99184, accuracy=97.61% +/- 1.38
position / eight: params=98831, accuracy=98.31% +/- 0.75
position / twelve: params=99328, accuracy=96.91% +/- 2.98
position / transformer: params=99056, accuracy=100.00% +/- 0.00
single / six: params=99184, accuracy=29.07% +/- 0.80
single / eight: params=98831, accuracy=28.32% +/- 0.21
single / twelve: params=99328, accuracy=24.92% +/- 0.58
single / transformer: params=99056, accuracy=28.77% +/- 0.66
multi / six: params=99184, accuracy=34.36% +/- 0.33
multi / eight: params=98831, accuracy=34.32% +/- 0.24
multi / twelve: params=99328, accuracy=32.68% +/- 0.08
multi / transformer: params=99056, accuracy=80.21% +/- 34.18
language / six: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / eight: params=94647, accuracy=42.05% +/- 0.97, perplexity=7.55 +/- 0.15
language / twelve: params=95733, accuracy=39.94% +/- 0.42, perplexity=8.19 +/- 0.31
language / transformer: params=96081, accuracy=43.16% +/- 0.46, perplexity=7.20 +/- 0.20

Reproduce with `depth_100k_test.py --architecture eight|twelve|transformer --seed 0|1|2`, then `summarize_depth_100k.py`. `check_depth_100k.py` verifies counts and five-step complete compiled model/Adam state and losses against ordinary updates for all new architectures. Two-block Transformer uses the previously validated full-first-block/selected-last-block implementation, now FF200 for both language and queries. This tests depth allocation at a fixed budget; it does not hold hidden width constant. No automatic paper or default-architecture changes made.

## Residuals in6/8/12-layer pooled models at approximately100k

Residual ablation in our pooled model only: approximately100k total parameters.
d64, learned token/position embeddings, GELU, non-learned pooled norm epsilon1e-5, no gate/final norm.
Depth/hidden width:6/120,8/103,12/84, identical across tasks.
Residual rule: h <- h + GELU(W h + b), at the K-1 equal-width hidden transitions; no residual on input/output projections, no added norm or scaling.
Residuals add zero parameters. Each residual configuration starts with exactly the same learned weights as its prior plain control.
Three seeds,10000 compiled Adam updates,batch64,lr .001,same training/evaluation data. Final checkpoint only.
Historical plain controls reused. Context32 language; context16 retrieval. No Transformer runs or speed measurements.

position / six-plain: params=99184, accuracy=97.61% +/- 1.38
position / six-residual: params=99184, accuracy=97.94% +/- 0.45
position / eight-plain: params=98831, accuracy=98.31% +/- 0.75
position / eight-residual: params=98831, accuracy=95.65% +/- 2.38
position / twelve-plain: params=99328, accuracy=96.91% +/- 2.98
position / twelve-residual: params=99328, accuracy=95.84% +/- 4.08
single / six-plain: params=99184, accuracy=29.07% +/- 0.80
single / six-residual: params=99184, accuracy=29.67% +/- 0.44
single / eight-plain: params=98831, accuracy=28.32% +/- 0.21
single / eight-residual: params=98831, accuracy=29.62% +/- 0.13
single / twelve-plain: params=99328, accuracy=24.92% +/- 0.58
single / twelve-residual: params=99328, accuracy=29.43% +/- 0.51
multi / six-plain: params=99184, accuracy=34.36% +/- 0.33
multi / six-residual: params=99184, accuracy=34.54% +/- 0.11
multi / eight-plain: params=98831, accuracy=34.32% +/- 0.24
multi / eight-residual: params=98831, accuracy=34.67% +/- 0.05
multi / twelve-plain: params=99328, accuracy=32.68% +/- 0.08
multi / twelve-residual: params=99328, accuracy=34.40% +/- 0.09
language / six-plain: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / six-residual: params=94473, accuracy=43.76% +/- 0.20, perplexity=7.37 +/- 0.12
language / eight-plain: params=94647, accuracy=42.05% +/- 0.97, perplexity=7.55 +/- 0.15
language / eight-residual: params=94647, accuracy=43.76% +/- 0.37, perplexity=7.51 +/- 0.30
language / twelve-plain: params=95733, accuracy=39.94% +/- 0.42, perplexity=8.19 +/- 0.31
language / twelve-residual: params=95733, accuracy=43.57% +/- 0.25, perplexity=7.38 +/- 0.10

Reproduce via `residual_depth_100k_test.py --architecture six|eight|twelve --seed 0|1|2`, then `summarize_residual_depth_100k.py`. `check_residual_depth_100k.py` verifies identical shared initial parameters, unchanged counts, and five-step compiled versus ordinary losses and full model/Adam state for every depth. Internal configuration fixed across language/query tasks; all trainable parameters counted. No automatic default-model or paper change.

## Six-layer wider embeddings under100k parameters

Six-layer pooling, wider embedding dimensions at <=100k total parameters.
Learned token/position embeddings, GELU, non-learned pooled norm epsilon1e-5, no residual/gate/final norm.
Embedding/hidden widths:64/120,128/109,256/88,512/49. Same shapes across query/language tasks.
Widths chosen to maximize hidden width under the query budget; language counts differ because vocabulary/context differ.
Three seeds,10000 compiled Adam updates,batch64,lr .001,same training/evaluation samples. Final checkpoint only.
Prior d64 controls reused. Context32 language; context16 retrieval. No speed test or new Transformer runs.

position / d64 h120: params=99184, accuracy=97.61% +/- 1.38
position / d128 h109: params=98907, accuracy=98.69% +/- 0.49
position / d256 h88: params=98992, accuracy=99.06% +/- 0.12
position / d512 h49: params=99531, accuracy=96.25% +/- 3.27
single / d64 h120: params=99184, accuracy=29.07% +/- 0.80
single / d128 h109: params=98907, accuracy=29.13% +/- 0.70
single / d256 h88: params=98992, accuracy=28.74% +/- 0.62
single / d512 h49: params=99531, accuracy=28.39% +/- 0.56
multi / d64 h120: params=99184, accuracy=34.36% +/- 0.33
multi / d128 h109: params=98907, accuracy=34.41% +/- 0.37
multi / d256 h88: params=98992, accuracy=34.24% +/- 0.60
multi / d512 h49: params=99531, accuracy=33.88% +/- 0.14
language / d64 h120: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / d128 h109: params=93577, accuracy=42.94% +/- 0.64, perplexity=7.49 +/- 0.05
language / d256 h88: params=92393, accuracy=42.48% +/- 0.28, perplexity=7.72 +/- 0.36
language / d512 h49: params=90301, accuracy=41.24% +/- 0.26, perplexity=8.24 +/- 0.14

Reproduce using `width_100k_test.py --architecture w128|w256|w512 --seed 0|1|2`, then `summarize_width_100k.py`. `check_width_100k.py` verifies total parameter formula/count and five-step complete compiled losses/model/Adam-state parity at each width. This reallocates a fixed total budget between embeddings and hidden layers; it is not an increase of d at fixed hidden width. No automatic paper/default-architecture changes made.

## Parallel shallow FFNs with equal averaging of probabilities

Shared d64 position-token pool followed by parallel one-hidden FFNs.
Each branch: Linear64->h, GELU, Linearh->V, softmax. Final distribution is equal1/N average of probabilities.
Shared learned token/position tables, non-learned pooled norm epsilon1e-5; no residual or final norm.
Branch count/hidden width:1/576,2/287,4/143,8/71. All trainable branch weights, biases and shared embeddings counted.
Same internal shapes across tasks. Query budget<=100k; language counts lower due to smaller vocabulary/context.
Three seeds,10000 compiled Adam updates,batch64,lr .001,same data/evaluation as prior runs. Final checkpoints only.
Deep6 reference reused, d64 h120, no residual. Different shallow branch initialization layout; not identical initial weights across branch counts.
Context32 language;16 retrieval. No speed measurement.

position / deep6: params=99184, accuracy=97.61% +/- 1.38
position / n1: params=100000, accuracy=71.20% +/- 0.83
position / n2: params=99774, accuracy=67.33% +/- 0.48
position / n4: params=99644, accuracy=51.95% +/- 2.82
position / n8: params=99384, accuracy=33.92% +/- 5.26
single / deep6: params=99184, accuracy=29.07% +/- 0.80
single / n1: params=100000, accuracy=29.15% +/- 0.50
single / n2: params=99774, accuracy=29.70% +/- 0.28
single / n4: params=99644, accuracy=30.42% +/- 0.14
single / n8: params=99384, accuracy=30.69% +/- 0.22
multi / deep6: params=99184, accuracy=34.36% +/- 0.33
multi / n1: params=100000, accuracy=34.46% +/- 0.13
multi / n2: params=99774, accuracy=34.61% +/- 0.15
multi / n4: params=99644, accuracy=34.69% +/- 0.17
multi / n8: params=99384, accuracy=34.71% +/- 0.12
language / deep6: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / n1: params=81153, accuracy=43.47% +/- 0.12, perplexity=7.45 +/- 0.12
language / n2: params=80958, accuracy=43.18% +/- 0.20, perplexity=7.31 +/- 0.11
language / n4: params=80828, accuracy=42.17% +/- 0.35, perplexity=7.20 +/- 0.14
language / n8: params=80568, accuracy=41.23% +/- 0.33, perplexity=7.20 +/- 0.17

Reproduce via `parallel_mixture_100k.py --architecture n1|n2|n4|n8 --seed 0|1|2`, then `summarize_parallel_mixture.py`. `check_parallel_mixture.py` checks parameter counts, arithmetic-mean probabilities and unit total mass, finite gradients for every branch, and five-step complete compiled-vs-ordinary model/Adam losses/state parity. Stable implementation uses logsumexp over branch log-softmax values minus logN, returning normalized log probabilities to the common CE interface. This is a probability mixture, not the mathematically equivalent wide shallow logit-sum architecture initially discussed. No automatic paper or default-architecture changes made.

## Parallel six-layer FFNs with equal averaging of probabilities

Shared d64 position-token pool followed by parallel six-hidden-layer FFNs.
Each branch: Linear64->h+GELU, five Linearh->h+GELU, Linearh->V, softmax. Final distribution is equal1/N average of probabilities.
Shared learned token/position tables, non-learned pooled norm epsilon1e-5; no residual or final norm.
Branch count/hidden width:1/120,2/81,4/53,8/34. All trainable branch weights, biases and shared embeddings counted.
Same internal shapes across tasks. Query budget<=100k; language counts lower due to smaller vocabulary/context.
Three seeds,10000 compiled Adam updates,batch64,lr .001,same data/evaluation as prior runs. Final checkpoints only.
Deep6 reference reused, d64 h120, no residual. Fresh one-branch six-layer control also rerun; parameter initialization layout differs from historical deep6; not identical initial weights across branch counts.
Context32 language;16 retrieval. No speed measurement.

position / deep6: params=99184, accuracy=97.61% +/- 1.38
position / n1: params=99184, accuracy=94.29% +/- 1.60
position / n2: params=99862, accuracy=52.79% +/- 4.06
position / n4: params=98924, accuracy=34.02% +/- 1.84
position / n8: params=99328, accuracy=23.20% +/- 0.21
single / deep6: params=99184, accuracy=29.07% +/- 0.80
single / n1: params=99184, accuracy=29.10% +/- 0.82
single / n2: params=99862, accuracy=27.86% +/- 0.29
single / n4: params=98924, accuracy=28.40% +/- 0.52
single / n8: params=99328, accuracy=28.67% +/- 0.46
multi / deep6: params=99184, accuracy=34.36% +/- 0.33
multi / n1: params=99184, accuracy=34.49% +/- 0.21
multi / n2: params=99862, accuracy=33.41% +/- 0.27
multi / n4: params=98924, accuracy=34.01% +/- 0.31
multi / n8: params=99328, accuracy=33.99% +/- 0.10
language / deep6: params=94473, accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / n1: params=94473, accuracy=43.44% +/- 0.39, perplexity=7.49 +/- 0.19
language / n2: params=93818, accuracy=42.50% +/- 0.69, perplexity=7.60 +/- 0.08
language / n4: params=91268, accuracy=41.22% +/- 0.51, perplexity=7.49 +/- 0.02
language / n8: params=89688, accuracy=39.53% +/- 0.32, perplexity=8.00 +/- 0.22

Reproduce via `deep_parallel_mixture_100k.py --architecture n1|n2|n4|n8 --seed 0|1|2`, then `summarize_deep_parallel_mixture.py`. `check_deep_parallel_mixture.py` checks parameter counts, arithmetic-mean probabilities and unit total mass, finite gradients for every branch, and five-step complete compiled-vs-ordinary model/Adam losses/state parity. Stable implementation uses logsumexp over branch log-softmax values minus logN, returning normalized log probabilities to the common CE interface. This is a probability mixture, not the mathematically equivalent wide shallow logit-sum architecture initially discussed. No automatic paper or default-architecture changes made.

## Original six-layer pooling: longer training

Longer training, only original pooled model: d64,h120,six hidden layers.
GELU,learned token/position embeddings,parameter-free pooled norm,no residual/gate/final norm.
Query parameters99184; language94473. Three seeds,Adam .001,batch64,no schedule/weight decay.
Train from the original initialization; the first10k updates reproduce prior loss/accuracy for every task/seed.
Checkpoints10k/30k/50k are predeclared; no best-step selection. Same validation/test examples as previous experiments.
Data generated in1000-update chunks, continuous Adam/model state throughout. Only weight files are saved to disk.
The50k pooled runs use five times the updates/examples of the prior10k Transformer runs; this is not a matched-training Transformer comparison. No Transformer runs or speed benchmark.

position / 10000 updates: accuracy=97.61% +/- 1.38
position / 30000 updates: accuracy=99.68% +/- 0.11
position / 50000 updates: accuracy=99.24% +/- 0.26
single / 10000 updates: accuracy=29.07% +/- 0.80
single / 30000 updates: accuracy=30.32% +/- 0.13
single / 50000 updates: accuracy=29.89% +/- 0.60
multi / 10000 updates: accuracy=34.36% +/- 0.33
multi / 30000 updates: accuracy=34.57% +/- 0.11
multi / 50000 updates: accuracy=34.50% +/- 0.15
language / 10000 updates: accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / 30000 updates: accuracy=45.56% +/- 0.49, perplexity=7.42 +/- 0.07
language / 50000 updates: accuracy=46.73% +/- 0.32, perplexity=7.38 +/- 0.27

Reproduce with `longer_training_100k.py --seed 0|1|2`, then `summarize_longer_training.py`. Predeclared50k updates and data RNG streams match the original first10k batches. All model parameters and Adam accumulators carry between scans, with no checkpoint restart/reset. JSON progress is saved each1000-update chunk; NPZ files at10k/30k/50k contain model weights only. No paper/default-model changes made.

## Longer training of the single-hidden-layer control

Longer training: one hidden layer d64 h576 versus six layers d64 h120.
Both: learned positions,GELU,parameter-free pooled norm,no residual/gate/final norm.
Shallow parameters100000 queries/81153 language; deep99184 queries/94473 language.
Three seeds,Adam .001,batch64,no schedule/weight decay. Continuous model/Adam state in1000-update scans.
First10k reproduces previous model-specific loss/accuracy in every task/seed. Same generated streams,held-out samples and language splits.
Checkpoints10k/30k/50k predeclared; no best-checkpoint selection. Deep50k results reused; only shallow newly trained.
Different hidden shape and initialization layout; language parameter counts not exactly matched. No Transformer runs or speed measurements.

position / shallow / 10000 updates: accuracy=71.20% +/- 0.83
position / shallow / 30000 updates: accuracy=82.90% +/- 3.92
position / shallow / 50000 updates: accuracy=88.83% +/- 3.86
position / deep / 10000 updates: accuracy=97.61% +/- 1.38
position / deep / 30000 updates: accuracy=99.68% +/- 0.11
position / deep / 50000 updates: accuracy=99.24% +/- 0.26
single / shallow / 10000 updates: accuracy=29.15% +/- 0.50
single / shallow / 30000 updates: accuracy=45.62% +/- 0.25
single / shallow / 50000 updates: accuracy=46.07% +/- 0.32
single / deep / 10000 updates: accuracy=29.07% +/- 0.80
single / deep / 30000 updates: accuracy=30.32% +/- 0.13
single / deep / 50000 updates: accuracy=29.89% +/- 0.60
multi / shallow / 10000 updates: accuracy=34.46% +/- 0.13
multi / shallow / 30000 updates: accuracy=34.65% +/- 0.11
multi / shallow / 50000 updates: accuracy=34.63% +/- 0.08
multi / deep / 10000 updates: accuracy=34.36% +/- 0.33
multi / deep / 30000 updates: accuracy=34.57% +/- 0.11
multi / deep / 50000 updates: accuracy=34.50% +/- 0.15
language / shallow / 10000 updates: accuracy=43.47% +/- 0.12, perplexity=7.45 +/- 0.12
language / shallow / 30000 updates: accuracy=46.75% +/- 0.38, perplexity=7.44 +/- 0.25
language / shallow / 50000 updates: accuracy=47.09% +/- 0.30, perplexity=7.72 +/- 0.24
language / deep / 10000 updates: accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / deep / 30000 updates: accuracy=45.56% +/- 0.49, perplexity=7.42 +/- 0.07
language / deep / 50000 updates: accuracy=46.73% +/- 0.32, perplexity=7.38 +/- 0.27

Reproduce via `longer_training_100k.py --architecture shallow --seed 0|1|2`, then `summarize_longer_shallow.py`. The shallow implementation is the earlier n1 control in `parallel_mixture_100k.py`, with one branch and one hidden layer (h576). Original deep long-run commands remain supported with the default architecture deep. All12 new10k model/task/seed baseline checks must pass. Saved NPZ checkpoints contain weights only, not optimizer state; Adam is uninterrupted in memory. No paper/default-model changes made.

## Original sigmoid without normalization, longer training

Original sigmoid/no-normalization versus GELU/parameter-free pooled normalization.
One hidden layer,d64,h576,learned positions,no residual/gate/final norm.
Same initial parameters,data streams,held-out examples and parameter counts:100000 queries/81153 language.
Three seeds,Adam .001,batch64,1000-update compiled chunks,continuous model/Adam state,no schedule/weight decay.
Predeclared checkpoints10k/30k/50k; no best-step selection. Only sigmoid/no-norm newly trained; GELU/norm controls reused.
Activation and pooled norm both change; this does not isolate either effect separately. No speed benchmark or Transformer runs.

position / sigmoid_no_norm / 10000 updates: accuracy=18.72% +/- 0.58
position / sigmoid_no_norm / 30000 updates: accuracy=38.53% +/- 0.56
position / sigmoid_no_norm / 50000 updates: accuracy=57.04% +/- 0.44
position / gelu_norm / 10000 updates: accuracy=71.20% +/- 0.83
position / gelu_norm / 30000 updates: accuracy=82.90% +/- 3.92
position / gelu_norm / 50000 updates: accuracy=88.83% +/- 3.86
single / sigmoid_no_norm / 10000 updates: accuracy=30.52% +/- 0.23
single / sigmoid_no_norm / 30000 updates: accuracy=30.80% +/- 0.25
single / sigmoid_no_norm / 50000 updates: accuracy=30.46% +/- 0.02
single / gelu_norm / 10000 updates: accuracy=29.15% +/- 0.50
single / gelu_norm / 30000 updates: accuracy=45.62% +/- 0.25
single / gelu_norm / 50000 updates: accuracy=46.07% +/- 0.32
multi / sigmoid_no_norm / 10000 updates: accuracy=34.80% +/- 0.17
multi / sigmoid_no_norm / 30000 updates: accuracy=34.77% +/- 0.17
multi / sigmoid_no_norm / 50000 updates: accuracy=34.87% +/- 0.20
multi / gelu_norm / 10000 updates: accuracy=34.46% +/- 0.13
multi / gelu_norm / 30000 updates: accuracy=34.65% +/- 0.11
multi / gelu_norm / 50000 updates: accuracy=34.63% +/- 0.08
language / sigmoid_no_norm / 10000 updates: accuracy=39.96% +/- 0.44, perplexity=7.82 +/- 0.04
language / sigmoid_no_norm / 30000 updates: accuracy=46.03% +/- 0.23, perplexity=7.08 +/- 0.14
language / sigmoid_no_norm / 50000 updates: accuracy=47.41% +/- 0.09, perplexity=7.34 +/- 0.16
language / gelu_norm / 10000 updates: accuracy=43.47% +/- 0.12, perplexity=7.45 +/- 0.12
language / gelu_norm / 30000 updates: accuracy=46.75% +/- 0.38, perplexity=7.44 +/- 0.25
language / gelu_norm / 50000 updates: accuracy=47.09% +/- 0.30, perplexity=7.72 +/- 0.24

Reproduce with `longer_training_100k.py --architecture sigmoid --seed 0|1|2`, then `summarize_longer_sigmoid.py`. `sigmoid_no_norm.py` implements raw position-token sum pooling, Linear, sigmoid, and output Linear with the exact same parameter layout/initial values as the earlier n1 shallow control. `check_sigmoid_no_norm.py` verifies identical weights/counts, expected unnormalized sigmoid formula, and five-step compiled versus ordinary full losses/model/Adam state for query and language shapes. Saved checkpoints contain model weights only; optimizer state remains continuous in memory. No paper/default changes made.

## Default two-block Transformer, matched longer training

Matched10k/30k/50k update comparison, same data streams and held-out samples within each task/seed.
Transformer:2 standard blocks,d64,4 heads,FF200 per block,learned positions,standard learned LayerNorm/residuals;99056 query/96081 language params.
Deep pooling:d64,h120,6 hidden layers,GELU+non-learned pooled norm;99184/94473 params.
Shallow pooling:d64,h576,1 hidden layer,GELU+non-learned pooled norm;100000/81153 params.
Sigmoid pooling:same shallow shape and counts,no normalization,sigmoid.
Three seeds,Adam .001,batch64,no schedule/weight decay,continuous model/Adam state in1000-update scans.
Only Transformer newly trained; pooled results reused. All predeclared milestones reported; no best-step selection.
Parameter budgets approximate; shallow language variants have fewer params. Same updates/examples are not equal compute. No new speed benchmark.

position / transformer / 10000 updates: accuracy=100.00% +/- 0.00
position / transformer / 30000 updates: accuracy=100.00% +/- 0.00
position / transformer / 50000 updates: accuracy=100.00% +/- 0.00
position / deep / 10000 updates: accuracy=97.61% +/- 1.38
position / deep / 30000 updates: accuracy=99.68% +/- 0.11
position / deep / 50000 updates: accuracy=99.24% +/- 0.26
position / shallow / 10000 updates: accuracy=71.20% +/- 0.83
position / shallow / 30000 updates: accuracy=82.90% +/- 3.92
position / shallow / 50000 updates: accuracy=88.83% +/- 3.86
position / sigmoid / 10000 updates: accuracy=18.72% +/- 0.58
position / sigmoid / 30000 updates: accuracy=38.53% +/- 0.56
position / sigmoid / 50000 updates: accuracy=57.04% +/- 0.44
single / transformer / 10000 updates: accuracy=28.77% +/- 0.66
single / transformer / 30000 updates: accuracy=30.09% +/- 0.31
single / transformer / 50000 updates: accuracy=29.99% +/- 0.31
single / deep / 10000 updates: accuracy=29.07% +/- 0.80
single / deep / 30000 updates: accuracy=30.32% +/- 0.13
single / deep / 50000 updates: accuracy=29.89% +/- 0.60
single / shallow / 10000 updates: accuracy=29.15% +/- 0.50
single / shallow / 30000 updates: accuracy=45.62% +/- 0.25
single / shallow / 50000 updates: accuracy=46.07% +/- 0.32
single / sigmoid / 10000 updates: accuracy=30.52% +/- 0.23
single / sigmoid / 30000 updates: accuracy=30.80% +/- 0.25
single / sigmoid / 50000 updates: accuracy=30.46% +/- 0.02
multi / transformer / 10000 updates: accuracy=80.21% +/- 34.18
multi / transformer / 30000 updates: accuracy=100.00% +/- 0.00
multi / transformer / 50000 updates: accuracy=100.00% +/- 0.00
multi / deep / 10000 updates: accuracy=34.36% +/- 0.33
multi / deep / 30000 updates: accuracy=34.57% +/- 0.11
multi / deep / 50000 updates: accuracy=34.50% +/- 0.15
multi / shallow / 10000 updates: accuracy=34.46% +/- 0.13
multi / shallow / 30000 updates: accuracy=34.65% +/- 0.11
multi / shallow / 50000 updates: accuracy=34.63% +/- 0.08
multi / sigmoid / 10000 updates: accuracy=34.80% +/- 0.17
multi / sigmoid / 30000 updates: accuracy=34.77% +/- 0.17
multi / sigmoid / 50000 updates: accuracy=34.87% +/- 0.20
language / transformer / 10000 updates: accuracy=43.16% +/- 0.46, perplexity=7.20 +/- 0.20
language / transformer / 30000 updates: accuracy=46.07% +/- 0.16, perplexity=6.77 +/- 0.05
language / transformer / 50000 updates: accuracy=46.72% +/- 0.76, perplexity=6.81 +/- 0.11
language / deep / 10000 updates: accuracy=43.25% +/- 0.38, perplexity=7.50 +/- 0.15
language / deep / 30000 updates: accuracy=45.56% +/- 0.49, perplexity=7.42 +/- 0.07
language / deep / 50000 updates: accuracy=46.73% +/- 0.32, perplexity=7.38 +/- 0.27
language / shallow / 10000 updates: accuracy=43.47% +/- 0.12, perplexity=7.45 +/- 0.12
language / shallow / 30000 updates: accuracy=46.75% +/- 0.38, perplexity=7.44 +/- 0.25
language / shallow / 50000 updates: accuracy=47.09% +/- 0.30, perplexity=7.72 +/- 0.24
language / sigmoid / 10000 updates: accuracy=39.96% +/- 0.44, perplexity=7.82 +/- 0.04
language / sigmoid / 30000 updates: accuracy=46.03% +/- 0.23, perplexity=7.08 +/- 0.14
language / sigmoid / 50000 updates: accuracy=47.41% +/- 0.09, perplexity=7.34 +/- 0.16

Reproduce with `longer_training_100k.py --architecture transformer --seed 0|1|2`, then `summarize_longer_transformer.py`. Uses the previously validated two-block implementation from `depth_100k_test.py`, computing all causal states in block1 and only supervised outputs in block2. The10k checkpoint reproduces its historical model/task/seed loss and accuracy before continuing. Models are freshly initialized, and optimizer state never resets within a run. Saved NPZ milestones contain model weights only.
