"""Exercise mode callbacks without DDS or physical vehicle output."""
from contextlib import ExitStack
import math
import unittest
from unittest.mock import Mock, patch

import test_mode_startup as startup


@unittest.skipIf(startup.ModeControlNode is None, 'ROS required')
class ManualTakeoverTests(unittest.TestCase):
    def setUp(self):
        from rcl_interfaces.msg import Parameter, ParameterValue
        from rcl_interfaces.srv import SetParametersAtomically
        self.service = SetParametersAtomically
        self.request = SetParametersAtomically.Request(parameters=[
            Parameter(name='mode', value=ParameterValue(type=4, string_value='MANUAL'))])
        self.node = startup.ModeStartupTests().build('AUTO')
        self.node.manual_lidar_safety = True
        self.node.pub = Mock()
        self.node.features = object()
        self.node.pig_stop_gate = Mock()
        self.node.pig_stop_gate.reason.return_value = None
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.clock = stack.enter_context(patch('control_modes.time.monotonic', return_value=100.))
        self.fresh = stack.enter_context(patch.object(self.node, '_fresh', return_value=True))
        stack.enter_context(patch.object(self.node, 'publish_state'))
        self.safety = stack.enter_context(patch('control_modes.safe_command',
            side_effect=lambda command, *a, **kw: (command, False)))

    def takeover(self):
        return self.node.set_mode(self.request, self.service.Response())

    def command(self, v=.05, w=0.):
        from geometry_msgs.msg import Twist
        message = Twist()
        message.linear.x, message.angular.z = v, w
        self.node.on_manual(message)

    def test_takeover_in_each_active_route_stage(self):
        for state in ('MAIN', 'CENTER', 'ENTRY_LEFT', 'CROSS_REVERSE', 'EXIT_MAIN'):
            with self.subTest(state=state):
                self.node.mode = 'AUTO'
                self.node.controller.state = state
                self.assertTrue(self.takeover().result.successful)
                self.assertEqual(self.node.commanded, (0., 0.))
                self.assertTrue(self.node.needs_restart)
                self.command()
                self.assertEqual(self.node.commanded, (.05, 0.))

    def test_twenty_hz_commands_then_release(self):
        self.takeover()
        for i in range(20):
            self.clock.return_value = 100. + i * .05
            self.command()
            self.assertEqual(self.node.mode, 'MANUAL')
        self.command(0., 0.)
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_timeout_and_late_command_cannot_resume(self):
        self.takeover()
        self.command()
        self.clock.return_value = 100.6
        self.command()
        self.assertEqual(self.node.mode_reason, 'MANUAL_TIMEOUT')
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_limits_and_nonfinite_stop(self):
        self.takeover()
        self.command(10., -10.)
        self.assertEqual(self.node.commanded, (2., -.6))
        self.command(math.nan)
        self.assertEqual(self.node.mode_reason, 'INVALID_MANUAL_COMMAND')
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_stale_scan_refuses_takeover(self):
        self.fresh.return_value = False
        self.assertFalse(self.takeover().result.successful)
        self.assertEqual(self.node.mode, 'STOP')
        self.assertEqual(self.node.mode_reason, 'SCAN_NOT_READY')

    def test_bypass_allows_manual_without_scan_or_clearance(self):
        self.node.manual_lidar_safety = False
        self.fresh.return_value = False
        self.node.features = None
        self.safety.side_effect = AssertionError('Manual bypass must not check LiDAR')
        self.assertTrue(self.takeover().result.successful)
        self.command(.1)
        self.assertEqual(self.node.commanded, (.1, 0.))
        self.node.on_scan(None)
        self.node.check_scan()
        self.assertEqual(self.node.mode, 'MANUAL')
        self.clock.return_value = 100.6
        self.node.check_scan()
        self.assertEqual(self.node.mode_reason, 'MANUAL_TIMEOUT')
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_bypass_preserves_external_stop_and_limits(self):
        self.node.manual_lidar_safety = False
        self.takeover()
        self.command(10., -10.)
        self.assertEqual(self.node.commanded, (2., -.6))

    def test_manual_speed_in_both_directions_preserves_auto_limit(self):
        self.node.manual_lidar_safety = False
        self.takeover()
        for requested, expected in ((.5, .5), (-.5, -.5), (2., 2.), (-2., -2.), (3., 2.), (-3., -2.)):
            self.command(requested)
            self.assertEqual(self.node.commanded, (expected, 0.))
        self.assertEqual(self.node.max_linear, .2)
        self.node.pig_stop_gate.reason.return_value = 'STOP'
        self.command()
        self.assertEqual(self.node.mode_reason, 'EXTERNAL_STOP')
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_manual_bypass_does_not_bypass_auto_scan_gate(self):
        self.node.manual_lidar_safety = False
        self.node.mode = 'STOP'
        self.fresh.return_value = False
        self.request.parameters[0].value.string_value = 'AUTO'
        self.assertFalse(self.takeover().result.successful)
        self.assertEqual(self.node.mode_reason, 'SCAN_NOT_READY')

    def test_external_stop_blocks_manual_motion(self):
        self.takeover()
        self.node.pig_stop_gate.reason.return_value = 'STOP'
        self.command()
        self.assertEqual(self.node.mode_reason, 'EXTERNAL_STOP')
        self.assertEqual(self.node.commanded, (0., 0.))

    def test_auto_requires_reset_after_takeover(self):
        self.takeover()
        self.request.parameters[0].value.string_value = 'AUTO'
        response = self.takeover()
        self.assertFalse(response.result.successful)
        self.assertEqual(response.result.reason, 'RESET_AT_START_REQUIRED')
        self.assertEqual(self.node.mode, 'STOP')
