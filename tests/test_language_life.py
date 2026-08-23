import unittest

from whatisi.language_teacher import LanguageTeacher, ScriptedBackend, ACTIONS
from whatisi.language_life import LanguageLife, LifeConfig, Event, ByteEventEncoder, SRC_ACT, SRC_OBS


class TestLanguageLife(unittest.TestCase):
    def test_teacher_targets_are_in_catalog(self):
        t = LanguageTeacher(ScriptedBackend(), seed=1)
        for _ in range(20):
            a = t.choose_target(list(ACTIONS))
            self.assertIn(a, ACTIONS)
            self.assertTrue(t.utterance(a))

    def test_provenance_not_encoded_in_bytes(self):
        a = Event(SRC_ACT, "say red")
        o = Event(SRC_OBS, "say red")
        ids_a, src_a = ByteEventEncoder.encode([a], 100)
        ids_o, src_o = ByteEventEncoder.encode([o], 100)
        self.assertTrue((ids_a == ids_o).all())
        self.assertFalse((src_a == src_o).all())

    def test_tiny_run(self):
        cfg = LifeConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            ff=64,
            memory_dim=8,
            max_tokens=96,
            context_events=4,
            unroll=2,
        )
        life = LanguageLife(LanguageTeacher(ScriptedBackend(), seed=2), cfg, seed=2, device="cpu")
        for _ in range(4):
            row = life.turn(train=True)
        self.assertEqual(row["step"], 4)
        self.assertGreaterEqual(row["memory_norm"], 0.0)
        self.assertIn(row["action_semantic"], ACTIONS)


if __name__ == "__main__":
    unittest.main()
