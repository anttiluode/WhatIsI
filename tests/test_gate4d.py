import unittest

import numpy as np
import torch

from whatisi.gate4b import Config, Model
from whatisi.gate4d import (
    GenericBinderConfig,
    GenericLateBinder,
    EXTERNAL,
    SELF_PRODUCED,
    I_PRONOUN,
    YOU_PRONOUN,
    _all_query_rows,
)


class TestGate4D(unittest.TestCase):
    def test_self_I_hides_speaker_identity(self):
        c = Config()
        hidden = np.zeros((1, c.steps, c.memory_dim), dtype=np.float32)
        # Minimal synthetic data with four current agents and a stable causal shape.
        data = {
            "name": np.tile(np.array([[0, 1, 2, 3]]), (c.steps, 1))[None, ...],
            "shape": np.tile(np.array([[4, 5, 6, 7]]), (c.steps, 1))[None, ...],
            "self_shape": np.full((1, c.steps), 6),
        }
        buckets = _all_query_rows(hidden, data, c, 123)
        row = buckets[(SELF_PRODUCED, I_PRONOUN)][0]
        self.assertEqual(row[4], -1)
        self.assertEqual(row[5], -1)
        self.assertEqual(row[6], 2)

    def test_visible_roles_remain_legitimate_inputs(self):
        c = Config()
        hidden = np.zeros((1, c.steps, c.memory_dim), dtype=np.float32)
        data = {
            "name": np.tile(np.array([[0, 1, 2, 3]]), (c.steps, 1))[None, ...],
            "shape": np.tile(np.array([[4, 5, 6, 7]]), (c.steps, 1))[None, ...],
            "self_shape": np.full((1, c.steps), 6),
        }
        buckets = _all_query_rows(hidden, data, c, 456)
        ext_i = buckets[(EXTERNAL, I_PRONOUN)][0]
        ext_you = buckets[(EXTERNAL, YOU_PRONOUN)][0]
        self.assertGreaterEqual(ext_i[4], 0)
        self.assertEqual(ext_i[6], ext_i[4])
        self.assertGreaterEqual(ext_you[5], 0)
        self.assertEqual(ext_you[6], ext_you[5])

    def test_generic_binder_has_no_factorized_address_input(self):
        c = Config()
        bc = GenericBinderConfig(labels=16, updates=1, layers=1)
        binder = GenericLateBinder(c, bc)
        B = 3
        agent_h = torch.zeros(B, c.n_agents, c.d_model)
        memory = torch.zeros(B, c.memory_dim)
        source = torch.zeros(B, dtype=torch.long)
        pronoun = torch.zeros(B, dtype=torch.long)
        speaker = torch.full((B,), c.pool, dtype=torch.long)
        addressee = torch.full((B,), c.pool, dtype=torch.long)
        logits = binder(agent_h, memory, source, pronoun, speaker, addressee)
        self.assertEqual(tuple(logits.shape), (B, c.pool))


if __name__ == "__main__":
    unittest.main()
