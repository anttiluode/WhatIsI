import unittest

from whatisi.gate4b import Config, gen_life


class TestGate4BWorld(unittest.TestCase):
    def test_hidden_actor_world_invariants(self):
        cfg = Config()
        life = gen_life(1234, cfg)

        # No motor consequence target during silent windows.
        silent = ~life["active"]
        self.assertTrue((life["actor_name"][silent] == -100).all())

        # Surface identities can change while the causal body remains the same.
        t = cfg.surface_swap_step
        self.assertEqual(life["self_shape"][t - 1], life["self_shape"][t])

        # Control transfer changes the causal body by construction.
        t = cfg.transfer_step
        self.assertNotEqual(life["self_shape"][t - 1], life["self_shape"][t])

    def test_observation_order_is_not_fixed_identity(self):
        cfg = Config()
        life = gen_life(5678, cfg)
        # The first observed row is not a stable body slot across the life.
        first_shapes = life["shape"][:, 0]
        self.assertGreater(len(set(int(x) for x in first_shapes)), 1)


if __name__ == "__main__":
    unittest.main()
