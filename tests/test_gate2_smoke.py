import unittest
from whatisi.gate1 import SequenceConfig
from whatisi.gate2 import generate_query_sequences


class Gate2Smoke(unittest.TestCase):
    def test_query_target_is_owner_value(self):
        cfg = SequenceConfig(steps=12, transfer_step=8, silent_windows=((3, 5),))
        s, v, y, owner = generate_query_sequences(2, 4, cfg)
        for b in range(4):
            for t in range(12):
                self.assertAlmostEqual(float(y[b, t]), float(v[b, t, owner[b, t]]), places=6)


if __name__ == "__main__":
    unittest.main()
