"""Run the CPU Tiny Shakespeare context-length screen from the repository root."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import optax

from experiments.benchmark import batch, count, jax, nnx, np
from experiments.budget_sweep import hidden_for_budget
from experiments.depth_100k_test import Model as Transformer
from experiments.language_depth_test import evaluate
from experiments.last_token_probe import TokenInteraction
from experiments.speed_benchmark import compiled_training

MODELS = ("token_product", "token_product_raw_gelu", "transformer")
RATES = (0.0003, 0.001)
MILESTONES = (1000, 2000, 3000)
CHUNK = 250


def make_model(name, context, seed):
    if name == "transformer":
        return Transformer("language", seed, "transformer")
    if name == "token_product":
        return TokenInteraction(seed, name, 576, 1, context, 65, width=64)
    hidden = hidden_for_budget(context, 65, 128, 100000, pooled_width=256)
    return TokenInteraction(seed, name, hidden, 1, context, 65, width=128)


def run(name, context, seed, rate, steps, train, monitor, heldout=None):
    model = make_model(name, context, seed)
    rng = np.random.default_rng(seed)
    state, update = compiled_training(model, optax.adam(rate))

    def sample_chunk():
        targets = rng.integers(128, len(train), size=(CHUNK, 64))
        return jax.device_put(batch(train, targets.reshape(-1) - context, context)[0]
                              .reshape(CHUNK, 64, context)), jax.device_put(train[targets])

    xs, ys = sample_chunk()
    jax.block_until_ready((xs, ys))
    compiled = update.lower(state, xs, ys).compile()
    history = []
    train_seconds = 0.0
    for start in range(0, steps, CHUNK):
        before = perf_counter()
        if start:
            xs, ys = sample_chunk()
        state, losses = compiled(state, xs, ys)
        jax.block_until_ready((state, losses))
        train_seconds += perf_counter() - before
        step = start + CHUNK
        if step in MILESTONES:
            nnx.update(model, state[0])
            history.append(dict(step=step, train_seconds=train_seconds,
                                **evaluate(model, *monitor)))
    result = dict(model=name, context=context, seed=seed, lr=rate,
                  parameters=count(model), history=history)
    if heldout is not None:
        result["test"] = evaluate(model, *heldout)
        graph, params = nnx.split(model)

        @jax.jit
        def predict(state, tokens):
            return nnx.merge(graph, state)(tokens)

        inputs = [jax.device_put(row[None]) for row in heldout[0][:128]]
        jax.block_until_ready(inputs)
        predict(params, inputs[0]).block_until_ready()
        timings = []
        for _ in range(7):
            before = perf_counter()
            for tokens in inputs:
                predict(params, tokens).block_until_ready()
            timings.append((perf_counter() - before) * 1000 / len(inputs))
        result["inference_batch_one_ms"] = median(timings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (Path(__file__).resolve().parents[1] / "data/shakespeare.txt").read_bytes()
    text = raw.decode()
    enc = {char: i for i, char in enumerate(sorted(set(text)))}
    data = np.array([enc[char] for char in text], dtype=np.int32)
    train, val, test = np.split(data, [int(len(data) * .8), int(len(data) * .9)])
    rng = np.random.default_rng(8675309)
    targets = [rng.integers(128, len(part), size=4096) for part in (val, test)]
    report = dict(dataset_sha256=hashlib.sha256(raw).hexdigest(), batch_size=64,
                  rates=RATES, milestones=MILESTONES, tuning=[], runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(section, result):
        report[section].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(section, json.dumps(result), flush=True)

    for context in (32, 128):
        monitor, heldout = [batch(part, positions - context, context)
                            for part, positions in zip((val, test), targets)]
        rates = {}
        for name in MODELS:
            candidates = [run(name, context, 3, rate, 1000, train, monitor)
                          for rate in RATES]
            for result in candidates:
                record("tuning", result)
            best = min(candidates, key=lambda row: row["history"][-1]["loss"])
            if not math.isfinite(best["history"][-1]["loss"]):
                raise ValueError(f"nonfinite validation loss for {name}, context {context}")
            rates[name] = best["lr"]
        for seed in range(3):
            for name in MODELS:
                record("runs", run(name, context, seed, rates[name], MILESTONES[-1],
                                   train, monitor, heldout))


if __name__ == "__main__":
    main()
