import unittest

from structured_lora.routing import build_route_masks, masks_are_nested


class RoutingTest(unittest.TestCase):
    def test_ordered_prefix_is_cumulative(self):
        masks = {
            depth: build_route_masks("ordered_prefix", [depth]).forward[0]
            for depth in range(1, 5)
        }
        self.assertTrue(masks_are_nested(masks))
        self.assertEqual(masks[1], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(masks[4], (1.0, 1.0, 1.0, 1.0))

    def test_local_credit_preserves_forward_but_only_updates_new_group(self):
        masks = build_route_masks("ordered_prefix_local_credit", [3])
        self.assertEqual(masks.forward[0], (1.0, 1.0, 1.0, 0.0))
        self.assertEqual(masks.credit[0], (0.0, 0.0, 1.0, 0.0))

    def test_flat_oracle_is_one_hot(self):
        masks = build_route_masks("flat_oracle", [1, 4])
        self.assertEqual(masks.forward[0], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(masks.forward[1], (0.0, 0.0, 0.0, 1.0))

    def test_fixed_random_order_is_a_symmetry_control(self):
        order = (2, 0, 3, 1)
        masks = {
            depth: build_route_masks("random_group_order", [depth], group_order=order).forward[0]
            for depth in range(1, 5)
        }
        self.assertTrue(masks_are_nested(masks))
        self.assertEqual(masks[1], (0.0, 0.0, 1.0, 0.0))

    def test_non_nested_control_breaks_cumulative_chain(self):
        masks = {
            depth: build_route_masks("non_nested_random_masks", [depth]).forward[0]
            for depth in range(1, 5)
        }
        self.assertFalse(masks_are_nested(masks))

    def test_knockout_removes_forward_and_credit(self):
        masks = build_route_masks("ordered_prefix_local_credit", [3], knockout_group=2)
        self.assertEqual(masks.forward[0], (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(masks.credit[0], (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
