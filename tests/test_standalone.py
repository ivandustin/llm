"""Checkpoint compatibility and batching checks for the standalone package."""
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from position_token_pooling import Model, NormalizedModel, StackedModel
from readout_experiments import save_weights
from validate_layer_depth import load_weights


class StandaloneModels(unittest.TestCase):
    def test_batch_matches_individual_examples(self):
        tokens = jnp.array([[0, 1, 2, 3], [3, 2, 0, 1]])
        positions = jnp.broadcast_to(jnp.arange(4), tokens.shape)
        examples = jnp.stack([positions, tokens], axis=1)
        for kind in (Model, NormalizedModel, StackedModel):
            with self.subTest(model=kind.__name__, module=kind.__module__):
                model = kind([4, 8], 8, 8, 6, nnx.Rngs(0))
                actual = nnx.vmap(model)(examples)
                expected = jnp.stack([model(example) for example in examples])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(actual.shape, (2, 8))

    def test_saved_parameter_names_and_checkpoint_roundtrip(self):
        model = StackedModel([4, 8], 8, 8, 6, nnx.Rngs(3), hidden_layers=6)
        example = jnp.stack([jnp.arange(4), jnp.array([3, 1, 0, 2])])
        expected = model(example)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.npz'
            save_weights(model, path)
            with np.load(path) as weights:
                self.assertIn('fn/a/kernel', weights.files)
                self.assertIn('fn/b/kernel', weights.files)
                self.assertIn('embeds/embeds/0/embedding', weights.files)
                self.assertIn('embeds/embeds/1/embedding', weights.files)
            restored = StackedModel([4, 8], 8, 8, 6, nnx.Rngs(99), hidden_layers=6)
            load_weights(restored, path)
            np.testing.assert_array_equal(expected, restored(example))


if __name__ == '__main__':
    unittest.main()
