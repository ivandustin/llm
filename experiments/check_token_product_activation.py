"""Check conditioned pooling, matched initialization, and optimizer parity."""
import argparse
import optax
from experiments.benchmark import nnx, np, jax, jnp, count
from experiments.longer_training_100k import TokenProductModel
from experiments.readout_experiments import learn
from experiments.speed_benchmark import compiled_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', choices=['sigmoid', 'gelu'], default='sigmoid')
    parser.add_argument('--norm', action='store_true')
    args = parser.parse_args()
    activate = jax.nn.sigmoid if args.activation == 'sigmoid' else jax.nn.gelu
    for task in ['position', 'single', 'multi', 'language']:
        model = TokenProductModel(task, 0,
                                  sigmoid_no_norm=args.activation == 'sigmoid' and not args.norm,
                                  gelu_no_norm=args.activation == 'gelu' and not args.norm,
                                  sigmoid_norm=args.activation == 'sigmoid' and args.norm)
        control = TokenProductModel(task, 0)
        assert count(model) == count(control) == (81153 if task == 'language' else 100000)
        a = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
        b = dict(nnx.to_flat_state(nnx.state(control, nnx.Param)))
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        length = 32 if task == 'language' else 16
        x = jnp.array(np.random.default_rng(0).integers(0, 31, (4, length)))
        base = model.base.base
        def expected_prefix(tokens):
            position, token = base.embeds.embeds
            products = position(jnp.arange(tokens.shape[1] - 1)) * token(tokens[:, :-1])
            z = (products * token(tokens[:, -1])[:, None, :]).sum(axis=1)
            if args.norm:
                z = (z - z.mean(axis=-1, keepdims=True)) / jnp.sqrt(z.var(axis=-1, keepdims=True) + 1e-5)
            return base.fn.b(activate(base.fn.a(z)))
        ends = [11] if task == 'single' else [11, 13, 15] if task == 'multi' else [length]
        expected = jnp.stack([expected_prefix(x[:, :end]) for end in ends], axis=1)
        if task not in ['single', 'multi']:
            expected = expected[:, 0]
        np.testing.assert_allclose(model(x), expected, rtol=1e-5, atol=1e-6)
        if task in ['single', 'multi']:
            original = np.asarray(model(x))
            for index, end in enumerate(ends):
                changed = x.at[:, end:].set(95)
                np.testing.assert_array_equal(model(changed)[:, index], original[:, index])
        # Full-state parity matters for the new compiled-training architecture.
        if task in ['multi', 'language']:
            other = nnx.clone(model)
            tx = optax.adam(.001)
            optimizer = nnx.Optimizer(other, tx, wrt=nnx.Param)
            rng = np.random.default_rng(7)
            xs = jnp.array(rng.integers(0, 31, (5, 4, length)))
            ys = jnp.array(rng.integers(0, 16, (5, 4, 3) if task == 'multi' else (5, 4)))
            state, run = compiled_training(model, tx)
            state, losses = run(state, xs, ys)
            manual = [learn(other, optimizer, xx, yy) for xx, yy in zip(xs, ys)]
            np.testing.assert_allclose(losses, manual, rtol=1e-4, atol=1e-5)
            aa = jax.tree.leaves(state)
            bb = jax.tree.leaves(nnx.state((other, optimizer)))
            assert len(aa) == len(bb)
            for left, right in zip(aa, bb):
                np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-5)
        print('PASS:', task, args.activation, 'norm=' + str(args.norm), 'matched weights/count, pooling formula and applicable causal/optimizer checks', flush=True)


if __name__ == '__main__':
    main()
