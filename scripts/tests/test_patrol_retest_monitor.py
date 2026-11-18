"""Independent audit must reject wrong-side, wrong-junction and stale poses."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'patrol_retest_monitor', Path(__file__).resolve().parents[1] / 'patrol_retest_monitor.py')
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class IndependentEndpointAudit(unittest.TestCase):
    def test_all_six_visits_and_wrong_side(self):
        for index, y in enumerate((-6.18, 0., 6.18, 6.18, 0., -6.18)):
            x = 3.18 if index < 3 else -3.18
            with self.subTest(index=index):
                self.assertTrue(monitor.endpoint_check(index, dict(x=x, y=y), .1)['passed'])
                self.assertFalse(monitor.endpoint_check(index, dict(x=-x, y=y), .1)['passed'])
                self.assertFalse(monitor.endpoint_check(index, dict(x=x, y=y+6.18), .1)['passed'])

    def test_main_aisle_is_not_branch_endpoint(self):
        self.assertFalse(monitor.endpoint_check(0, dict(x=.1, y=-6.18), .1)['passed'])

    def test_stale_or_missing_pose_rejected(self):
        self.assertFalse(monitor.endpoint_check(0, None, .1)['passed'])
        self.assertFalse(monitor.endpoint_check(0, dict(x=3.18, y=-6.18), 2.1)['passed'])

    def test_pose_parser_accepts_protobuf_omitted_zero_fields(self):
        block = '\n  name: "wheeltec_mini"\n  position {\n    y: -6.18\n  }\n  orientation {\n    w: 1\n  }\n'
        pose = monitor.pose_record(block)
        self.assertEqual(pose, dict(x=0., y=-6.18, z=0., roll=0., pitch=0., yaw=0.))
        self.assertIsNone(monitor.pose_record(block.replace('wheeltec_mini', 'another_robot')))


if __name__ == '__main__':
    unittest.main()
