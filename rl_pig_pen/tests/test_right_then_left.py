"""LiDAR raycasts and ideal kinematics; no DDS, UART, odometry or real vehicle."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_patrol_right_then_left import RightThenLeftPatrolController
from lidar_patrol_controller import TraditionalPatrolController
from lidar_scene import StaticLidarScene
from junction_localization import JunctionTracker
from navigation import collision_detected, load_config, scan_features, wrap_angle


class RightThenLeftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = StaticLidarScene()
        cls.cfg = replace(load_config(Path(__file__).resolve().parents[1] /
                                      'patrol_config_right_then_left.json'), persistent_recovery=False)

    def features(self, pose):
        return scan_features(self.scene.scan(pose), self.cfg)

    def test_entry_confirmation_corrects_residual_yaw_without_translation(self):
        c = RightThenLeftPatrolController(self.cfg)
        c.state, c.entry_phase = 'ENTRY_LEFT', 'WALL_CONFIRM'
        f = self.features((0., -8., math.pi / 2))
        with patch('lidar_patrol_right_then_left.incoming_corridor', return_value=(-.104, .103)):
            command = c._entry_wall_command(f, 1.)
        self.assertEqual(command[0], 0.)
        self.assertLess(command[1], 0.)
        self.assertLessEqual(abs(command[1]), .15)
        self.assertEqual(c.confirm, 0)
        self.assertEqual(c.state, 'ENTRY_LEFT')
        for corridor in (None, (.104, .5), (.5, .1)):
            with patch('lidar_patrol_right_then_left.incoming_corridor', return_value=corridor):
                np.testing.assert_array_equal(c._entry_wall_command(f, 2.), [0., 0.])

    def run_route(self, initial, *, horizon=None):
        cfg = self.cfg if horizon is None else replace(self.cfg, control_dt=horizon)
        controller = RightThenLeftPatrolController(cfg)
        pose = np.array(initial, dtype=float)
        endpoints = []
        previous_endpoints = 0
        transitions = []
        for step in range(20000):
            features = self.features(pose)
            self.assertFalse(collision_detected(features, cfg), repr(pose))
            v, w = controller.command(features, step / 12.)
            state = (controller.pass_side, controller.state)
            if not transitions or transitions[-1] != state:
                transitions.append(state)
            self.assertNotEqual(controller.state, 'FAILED',
                (controller.failure, pose, controller.pass_side, controller.diagnostics))
            if controller.endpoints_reached != previous_endpoints:
                endpoints.append((controller.pass_side, pose.copy()))
                previous_endpoints = controller.endpoints_reached
            if controller.state == 'DONE':
                break
            if controller.state in ('TURN_LEFT', 'TURN_MAIN'):
                self.assertEqual(v, 0.)
            if controller.state in ('MAIN', 'FINAL_MAIN'):
                self.assertGreaterEqual(controller.main_direction * v, 0.)
            pose += np.array([v * math.cos(pose[2]), v * math.sin(pose[2]), w]) / 12.
        else:
            self.fail(('route did not finish', controller.pass_side,
                       controller.state, pose, controller.diagnostics))
        self.assertEqual(controller.side_endpoints, {'RIGHT': 3, 'LEFT': 3})
        self.assertEqual(controller.junctions_done, 6)
        self.assertEqual(controller.endpoints_reached, 6)
        self.assertEqual([side for side, _ in endpoints], ['RIGHT'] * 3 + ['LEFT'] * 3)
        self.assertEqual(controller.visits,
            [('RIGHT', 1), ('RIGHT', 2), ('RIGHT', 3),
             ('LEFT', 1), ('LEFT', 2), ('LEFT', 3)])
        # Independently verify physical side and crossing order using plant pose.
        self.assertTrue(all(p[0] > 2.5 for _, p in endpoints[:3]))
        self.assertTrue(all(p[0] < -2.5 for _, p in endpoints[3:]))
        self.assertTrue(all(a[1] < b[1] for (_, a), (_, b) in zip(endpoints[:2], endpoints[1:3])))
        self.assertTrue(all(a[1] > b[1] for (_, a), (_, b) in zip(endpoints[3:5], endpoints[4:])))
        self.assertLess(pose[1], -10.5)
        self.assertLess(abs(pose[0]), .10)
        self.assertLess(abs(wrap_angle(pose[2] - math.pi / 2)), .10)
        np.testing.assert_array_equal(controller.command(self.features(pose), step / 12. + .1), [0., 0.])
        return controller, pose, transitions

    def test_full_route_right_then_left_return_to_start(self):
        self.run_route(self.cfg.start)

    def test_full_route_with_scan_watchdog_horizon(self):
        self.run_route(self.cfg.start, horizon=.5)

    def test_full_route_with_offset_and_yaw_error(self):
        self.run_route((.06, -8., math.pi / 2 + .05))

    def test_premature_main_end_is_not_pass_completion(self):
        for side, pose in (('RIGHT', (0., 11.12, math.pi / 2)),
                           ('LEFT', (0., -11.12, math.pi / 2))):
            with self.subTest(side=side):
                c = RightThenLeftPatrolController(self.cfg)
                c.pass_side = side
                c.main_direction = 1 if side == 'RIGHT' else -1
                for i in range(4):
                    c.command(self.features(pose), i / 12.)
                self.assertEqual(c.failure, 'main_end_before_patrol_complete')
                self.assertEqual(c.endpoints_reached, 0)

    def test_repeated_scan_cannot_switch_pass(self):
        c = RightThenLeftPatrolController(self.cfg)
        c.state = 'FINAL_MAIN'
        c.pass_junctions_done = 3
        c.junctions_done = 3
        c.endpoints_reached = 3
        c.side_endpoints['RIGHT'] = 3
        features = self.features((0., 11.12, math.pi / 2))
        for _ in range(5):
            c.command(features, 1.)
        self.assertEqual(c.pass_side, 'RIGHT')
        c.command(features, 1.1)
        c.command(features, 1.2)
        c.command(features, 1.3)
        self.assertEqual(c.pass_side, 'LEFT')
        self.assertEqual(c.main_direction, -1)
        self.assertEqual(c.pass_junctions_done, 0)
        self.assertEqual(c.junctions_done, 3)

    def test_forward_and_reverse_contact_keep_obstacle_protection(self):
        for side, sign in (('RIGHT', 1), ('LEFT', -1)):
            c = RightThenLeftPatrolController(self.cfg)
            c.pass_side, c.main_direction = side, sign
            scan = self.scene.scan(self.cfg.start)
            angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
            delta = np.arctan2(np.sin(angles - (0. if sign > 0 else math.pi)),
                               np.cos(angles - (0. if sign > 0 else math.pi)))
            scan.ranges[np.abs(delta) < .05] = .20
            features = scan_features(scan, self.cfg)
            np.testing.assert_array_equal(c.command(features, 1.), [0., 0.])
            # Missing corridor geometry may hold before a drive command is
            # proposed; also verify the actual signed motion guard directly.
            np.testing.assert_array_equal(c._safe([sign * .2, 0.], features), [0., 0.])
            self.assertEqual(c.failure, 'obstacle_clearance')
            self.assertEqual(c.endpoints_reached, 0)

    def test_external_hold_preserves_pass_and_counters(self):
        c = RightThenLeftPatrolController(self.cfg)
        c.pass_side, c.main_direction = 'LEFT', -1
        c.junctions_done, c.pass_junctions_done = 4, 1
        c.pause_external()
        for i in range(4):
            np.testing.assert_array_equal(c.command(self.features((0., 8., math.pi / 2)), i / 12.), [0., 0.])
        self.assertEqual((c.pass_side, c.main_direction, c.junctions_done, c.pass_junctions_done),
                         ('LEFT', -1, 4, 1))

    def test_persistent_contact_wait_does_not_advance_route(self):
        cfg = replace(self.cfg, persistent_recovery=True)
        c = RightThenLeftPatrolController(cfg)
        scan = self.scene.scan(cfg.start)
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        scan.ranges[np.abs(angles) < .05] = .20
        features = scan_features(scan, cfg)
        c._safe([.2, 0.], features)
        for i in range(6):
            np.testing.assert_array_equal(c.command(features, i / 12.), [0., 0.])
        self.assertEqual(c.obstacle['phase'], 'WAIT')
        self.assertEqual(c.pass_side, 'RIGHT')
        self.assertEqual(c.side_endpoints, {'RIGHT': 0, 'LEFT': 0})
        self.assertEqual(c.junctions_done, 0)

    def test_scan_gap_or_reversed_time_cannot_advance_route(self):
        for stamp in (.5, 2.):
            c = RightThenLeftPatrolController(self.cfg)
            features = self.features(self.cfg.start)
            c.command(features, 1.)
            np.testing.assert_array_equal(c.command(features, stamp), [0., 0.])
            if stamp < 1.:
                self.assertEqual(c.state, 'FAILED')
            else:
                self.assertEqual(c.relocalization['reason'], 'scan_gap')
            self.assertEqual(c.endpoints_reached, 0)
            self.assertEqual(c.pass_side, 'RIGHT')

    def test_legacy_controller_keeps_original_route(self):
        c = TraditionalPatrolController(self.cfg)
        self.assertFalse(hasattr(c, 'pass_side'))
        self.assertEqual(c.state, 'MAIN')

    def test_turn_drift_recentres_before_retry_without_revisiting_branch(self):
        cases = (
            ('TURN_MAIN', (.155, -6.2, math.pi / 2 - .07), -math.pi / 2 + .07, 'EXIT_MAIN', 1),
            ('TURN_LEFT', (0., -6.355, .07), math.pi / 2 - .07, 'ENTRY_LEFT', 0),
        )
        for state, initial, incoming, target, endpoints in cases:
            with self.subTest(state=state):
                c = RightThenLeftPatrolController(self.cfg)
                c.state = state
                c.endpoints_reached = endpoints
                c.side_endpoints['RIGHT'] = endpoints
                pose = np.array(initial, dtype=float)
                features = self.features(pose)
                geometry = c._fresh_junction_geometry(features)
                self.assertIsNotNone(geometry)
                c.tracker = JunctionTracker(geometry, -.05, c.cfg, heading_hint=incoming)
                states = []
                for step in range(1200):
                    features = self.features(pose)
                    v, w = c.command(features, step / 12.)
                    states.append(c.state)
                    self.assertNotEqual(c.state, 'FAILED', (c.failure, c.diagnostics))
                    self.assertFalse(collision_detected(features, self.cfg))
                    if c.turn_reposition_attempts and c.state in ('TURN_MAIN', 'TURN_LEFT'):
                        self.assertLessEqual(abs(w), .25)
                    if c.state == target:
                        break
                    pose += np.array([v * math.cos(pose[2]), v * math.sin(pose[2]), w]) / 12.
                else:
                    self.fail((state, 'turn recovery did not finish', c.state, c.diagnostics))
                self.assertIn('RETURN_CENTER' if state == 'TURN_MAIN' else 'CENTER', states)
                self.assertGreater(c.turn_reposition_attempts, 0)
                self.assertEqual(c.endpoints_reached, endpoints)
                self.assertEqual(c.side_endpoints, {'RIGHT': endpoints, 'LEFT': 0})
                self.assertEqual(c.cfg.junction_turn_centre_limit, .15)

    def test_turn_reposition_retry_limit_stops_instead_of_looping(self):
        c = RightThenLeftPatrolController(self.cfg)
        c.state = 'TURN_MAIN'
        c.turn_reposition_attempts = 2
        c.relocalization = {'reason': 'turn_centre_drift', 'failed_state': 'TURN_MAIN'}
        features = self.features((.155, -6.2, math.pi / 2))
        geometry = c._fresh_junction_geometry(features)
        tracker = JunctionTracker(geometry, 1., c.cfg, heading_hint=-math.pi / 2)
        np.testing.assert_array_equal(c._finish_relocalization(1., tracker), [0., 0.])
        self.assertEqual(c.failure, 'turn_reposition_limit')
        self.assertEqual(c.state, 'FAILED')

    def test_failed_turn_recentering_is_terminal(self):
        for reason in ('centre_overshoot', 'centre_reposition_limit'):
            c = RightThenLeftPatrolController(self.cfg)
            c.state = 'RETURN_CENTER'
            c.turn_reposition_attempts = 1
            np.testing.assert_array_equal(c.fail(reason), [0., 0.])
            self.assertEqual(c.failure, 'turn_reposition_limit')
            self.assertEqual(c.state, 'FAILED')


if __name__ == '__main__':
    unittest.main()
