import tempfile
import unittest
from pathlib import Path

import torch

from whatisi.language_teacher import LanguageTeacher, ScriptedBackend, ACTIONS
from whatisi.language_life import LanguageLife, LifeConfig, Event, EventEncoder, SRC_ACT, SRC_OBS


class TestLanguageLife(unittest.TestCase):
    def tiny_cfg(self):
        return LifeConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            ff=64,
            memory_dim=8,
            max_tokens=64,
            context_events=4,
            unroll=2,
            consolidate_every=0,
            consolidation_steps=0,
        )

    def test_teacher_targets_are_in_catalog(self):
        t = LanguageTeacher(ScriptedBackend(), seed=1)
        for _ in range(20):
            a = t.choose_target(list(ACTIONS))
            self.assertIn(a, ACTIONS)
            self.assertTrue(t.utterance(a))

    def test_provenance_is_separate_from_word_tokens(self):
        a = Event(SRC_ACT, "touch red")
        o = Event(SRC_OBS, "touch red")
        ids_a, src_a = EventEncoder.encode([a], 100)
        ids_o, src_o = EventEncoder.encode([o], 100)
        self.assertTrue((ids_a == ids_o).all())
        self.assertFalse((src_a == src_o).all())

    def test_tiny_run(self):
        life = LanguageLife(LanguageTeacher(ScriptedBackend(), seed=2), self.tiny_cfg(), seed=2, device="cpu")
        for _ in range(4):
            row = life.turn(train=True)
        self.assertEqual(row["step"], 4)
        self.assertGreaterEqual(row["memory_norm"], 0.0)
        self.assertIn(row["action_semantic"], ACTIONS)

    def test_checkpoint_preserves_life_state_and_context(self):
        cfg = self.tiny_cfg()
        life = LanguageLife(LanguageTeacher(ScriptedBackend(), seed=3), cfg, seed=3, device="cpu")
        for _ in range(3):
            life.turn(train=True)
        expected_memory = life.memory.detach().clone()
        expected_step = life.step
        expected_events = [(e.source, e.text, e.changed) for e in life.events]

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "life.pt"
            life.checkpoint(path)
            restored = LanguageLife(LanguageTeacher(ScriptedBackend(), seed=3), cfg, seed=3, device="cpu")
            restored.load_checkpoint(path)

        self.assertEqual(restored.step, expected_step)
        self.assertTrue(torch.allclose(restored.memory, expected_memory))
        self.assertEqual([(e.source, e.text, e.changed) for e in restored.events], expected_events)


if __name__ == "__main__":
    unittest.main()
