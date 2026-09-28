import unittest

from scripts.spike012_execution import StopEdge
from scripts.analyze_spike012_stop_v2 import low_force_segments


class StopDiagnosticsTest(unittest.TestCase):
    def test_w1_edges_and_latched_mask(self):
        edge = StopEdge()
        timeline = []
        for second, moving in ((0, False), (1, True), (3, False), (5, True),
                               (6, True), (7, False)):
            for step in range(25):
                event, active, source = edge.update(moving)
                timeline.append((second + step*.04, event, active, source))
        events = [(t, source) for t, event, _, source in timeline if event]
        self.assertEqual(events, [(3., "request_nonzero_to_zero"), (7., "request_nonzero_to_zero")])
        self.assertFalse(any(active for t, _, active, _ in timeline if t < 3))
        self.assertTrue(all(active for t, _, active, _ in timeline if 3 <= t < 5))
        self.assertFalse(any(active for t, _, active, _ in timeline if 5 <= t < 7))

    def test_gate_reset_and_reverse_stop(self):
        edge = StopEdge()
        edge.update(True)
        self.assertEqual(edge.update(True, False), (True, True, "channel_closed_while_moving"))
        self.assertEqual(edge.update(True, False), (False, True, "none"))
        edge.reset()
        self.assertEqual(edge.update(False), (False, False, "none"))
        edge.update(True)
        self.assertEqual(edge.update(False), (True, True, "request_nonzero_to_zero"))
        edge.reset()
        edge.update(True)
        self.assertEqual(edge.update(False, False), (True, True, "channel_closed_on_stop"))

    def test_low_force_longest_and_window_truncation(self):
        one = low_force_segments([4, 4, 4, 6, 6, 6], dt=1)
        two = low_force_segments([4, 6, 4, 6, 4, 6], dt=1)
        self.assertEqual(one["cumulative_s"], two["cumulative_s"])
        self.assertEqual((one["longest_continuous_s"], two["longest_continuous_s"]), (3, 1))
        self.assertTrue(one["segments"][0]["left_truncated"])
        self.assertFalse(one["segments"][0]["right_truncated"])
        self.assertTrue(low_force_segments([6, 4, 4], dt=1)["segments"][0]["right_truncated"])


if __name__ == "__main__":
    unittest.main()
