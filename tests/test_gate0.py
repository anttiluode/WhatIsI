import unittest
import numpy as np
from whatisi.gate0 import CausalSelfPointer, run_gate0


class Gate0Tests(unittest.TestCase):
    def test_silent_update_preserves_state(self):
        p = CausalSelfPointer(4)
        p.update(1.0, np.array([3.0, 0.0, 0.0, 0.0]))
        before = p.score.copy()
        p.update(0.0, np.array([100.0, -100.0, 5.0, 2.0]))
        np.testing.assert_allclose(p.score, before)

    def test_gate0(self):
        result = run_gate0(seeds=range(100, 108))
        self.assertTrue(result["pass"], result)


if __name__ == "__main__":
    unittest.main()
