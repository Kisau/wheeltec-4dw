"""Mode adapter callbacks only; never construct DDS or command hardware."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation import load_config
from lidar_patrol_right_then_left import RightThenLeftPatrolController
try:
    import control_modes_right_then_left as modes
except ImportError:
    modes = None


@unittest.skipIf(modes is None, 'ROS required')
class RightThenLeftModeTests(unittest.TestCase):
    def build(self, initial):
        def base_init(n):
            n.cfg = replace(load_config(), junctions=3)
            n.controller = RightThenLeftPatrolController(n.cfg)
        def parameter(n, name, default):
            return SimpleNamespace(value=initial if name == 'initial_mode' else default)
        cls = modes.ModeControlNode
        with patch.object(modes.LidarPatrolNode, '__init__', base_init), \
             patch.object(cls, 'declare_parameter', parameter), \
             patch.object(cls, 'create_publisher', return_value=Mock()), \
             patch.object(cls, 'create_subscription', return_value=Mock()), \
             patch.object(cls, 'create_service', return_value=Mock()), \
             patch.object(cls, 'publish_state'), \
             patch.object(cls, 'get_logger', return_value=Mock()):
            n = cls()
        n.pub = Mock()
        return n

    def test_default_stop_and_initial_auto_use_new_route(self):
        for initial in ('STOP', 'AUTO'):
            with self.subTest(initial=initial):
                n = self.build(initial)
                self.assertEqual(n.mode, initial)
                self.assertIsInstance(n.controller, RightThenLeftPatrolController)
                self.assertEqual(n.controller.pass_side, 'RIGHT')
                self.assertTrue(n.controller.external_paused)
                self.assertEqual(n.commanded, (0., 0.))

    def test_manual_takeover_still_requires_route_reset(self):
        from rcl_interfaces.msg import Parameter, ParameterValue
        from rcl_interfaces.srv import SetParametersAtomically as Service
        n = self.build('AUTO')
        request = Service.Request(parameters=[
            Parameter(name='mode', value=ParameterValue(type=4, string_value='MANUAL'))])
        with patch.object(n, 'publish_state'):
            response = n.set_mode(request, Service.Response())
        self.assertTrue(response.result.successful)
        self.assertEqual(n.mode, 'MANUAL')
        self.assertTrue(n.needs_restart)
        self.assertEqual(n.commanded, (0., 0.))

    def test_reset_clears_both_passes_using_new_controller(self):
        from std_srvs.srv import Trigger
        n = self.build('STOP')
        n.controller.pass_side, n.controller.main_direction = 'LEFT', -1
        n.controller.junctions_done = 4
        n.controller.side_endpoints = {'RIGHT': 3, 'LEFT': 1}
        n.needs_restart = True
        n.features = object()
        with patch.object(n, '_fresh', return_value=True), \
             patch.object(n, 'publish_state'), \
             patch.object(modes, 'safe_command', return_value=((0., 0.), False)), \
             patch.object(modes, 'incoming_corridor', return_value=(0., 0.)):
            response = n.reset_at_start(Trigger.Request(), Trigger.Response())
        self.assertTrue(response.success)
        self.assertIsInstance(n.controller, RightThenLeftPatrolController)
        self.assertEqual(n.controller.pass_side, 'RIGHT')
        self.assertEqual(n.controller.side_endpoints, {'RIGHT': 0, 'LEFT': 0})
        self.assertEqual(n.controller.visits, [])
        self.assertEqual(n.mode, 'STOP')
        self.assertFalse(n.needs_restart)

    def test_status_identifies_pass_and_side_counts(self):
        import json
        n = self.build('STOP')
        n.controller.pass_side = 'LEFT'
        n.controller.side_endpoints = {'RIGHT': 3, 'LEFT': 1}
        with patch.object(modes.LidarPatrolNode, 'publish_state'):
            n.publish_state()
        payload = json.loads(n.control_pub.publish.call_args.args[0].data)
        self.assertEqual(payload['route_pass'], 'LEFT')
        self.assertEqual(payload['side_endpoints'], {'RIGHT': 3, 'LEFT': 1})


if __name__ == '__main__':
    unittest.main()
