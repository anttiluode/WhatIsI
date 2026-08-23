import unittest

import numpy as np
import torch

from whatisi.gate4b import Config, make
from whatisi.gate4c import (
    EXTERNAL,
    SELF_PRODUCED,
    I_PRONOUN,
    YOU_PRONOUN,
    LexicalRouter,
    make_late_queries,
)


class TestGate4CLateBinding(unittest.TestCase):
    def test_self_I_has_no_external_speaker_identity(self):
        cfg = Config()
        data = make(12345, 4, cfg)
        hidden = np.zeros((4, cfg.steps, cfg.memory_dim), dtype="float32")
        rows = make_late_queries(hidden, data, cfg, seed=77)
        self_i = [r for r in rows if r[2] == SELF_PRODUCED and r[3] == I_PRONOUN]
        self.assertGreater(len(self_i), 0)
        for life, t, source, pronoun, speaker, addressee, target in self_i:
            self.assertEqual(speaker, -1)
            self.assertEqual(addressee, -1)
            shape = int(data["self_shape"][life, t])
            slot = int(np.where(data["shape"][life, t] == shape)[0][0])
            self.assertEqual(target, int(data["name"][life, t, slot]))

    def test_visible_role_targets_are_not_causal_labels(self):
        cfg = Config()
        data = make(54321, 4, cfg)
        hidden = np.zeros((4, cfg.steps, cfg.memory_dim), dtype="float32")
        rows = make_late_queries(hidden, data, cfg, seed=88)
        for row in rows:
            source, pronoun, speaker, addressee, target = row[2:]
            if source == EXTERNAL and pronoun == I_PRONOUN:
                self.assertEqual(target, speaker)
            if pronoun == YOU_PRONOUN:
                self.assertEqual(target, addressee)

    def test_router_is_a_three_way_address_mixture(self):
        router = LexicalRouter()
        source = torch.tensor([EXTERNAL, SELF_PRODUCED])
        pronoun = torch.tensor([I_PRONOUN, YOU_PRONOUN])
        w = router.weights(source, pronoun)
        self.assertEqual(tuple(w.shape), (2, 3))
        self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
