import unittest
from whatisi.gate1 import SequenceConfig, generate_sequences, pointer_predictions


class Gate1Smoke(unittest.TestCase):
    def test_shapes(self):
        cfg = SequenceConfig(steps=20, transfer_step=12, silent_windows=((5, 7),))
        x, y = generate_sequences(1, 3, cfg)
        p = pointer_predictions(x, cfg)
        self.assertEqual(x.shape, (3, 20, 5))
        self.assertEqual(y.shape, (3, 20))
        self.assertEqual(p.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
