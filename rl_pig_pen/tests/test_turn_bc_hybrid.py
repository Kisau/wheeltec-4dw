"""Verify model ownership and retained guards without running Gazebo."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lidar_patrol_right_then_left import RightThenLeftPatrolController
from navigation import load_config


class TurnOwnership(unittest.TestCase):
    def controller(self, policy):
        cfg=load_config(Path(__file__).resolve().parents[1]/'patrol_config_right_then_left.json')
        c=RightThenLeftPatrolController(cfg,turn_policy=policy)
        c.state='TURN_LEFT'
        geometry=SimpleNamespace(centre=np.zeros(2))
        c.tracker=SimpleNamespace(incoming_axis=0.,geometry=geometry,stamp=1.)
        c._geometry=Mock(return_value=geometry)
        c._track_geometry=Mock(return_value=geometry)
        c._safe=Mock(side_effect=lambda action,features:np.asarray(action))
        return c

    def test_model_owns_angular_command_and_never_linear(self):
        policy=Mock(return_value=-.123);c=self.controller(policy)
        np.testing.assert_allclose(c.command(None,1.),[0.,-.123])
        policy.assert_called_once();c._safe.assert_called_once()
        self.assertEqual(c.diagnostics['command_owner'],'BC_TURN')

    def test_original_controller_unchanged_when_no_model(self):
        c=self.controller(None)
        np.testing.assert_allclose(c.command(None,1.),[0.,-.6])

    def test_nonfinite_model_stops_with_failure(self):
        c=self.controller(Mock(return_value=float('nan')))
        np.testing.assert_array_equal(c.command(None,1.),[0.,0.])
        self.assertEqual(c.failure,'bc_turn_policy_fault')

    def test_centre_drift_blocks_model(self):
        policy=Mock(return_value=-.3);c=self.controller(policy)
        c.tracker.geometry.centre[:]=.3
        np.testing.assert_array_equal(c.command(None,1.),[0.,0.])
        policy.assert_not_called()

    def test_external_stop_does_not_call_model(self):
        policy=Mock(return_value=-.3);c=self.controller(policy);c.external_paused=True
        c.hold_external=Mock(return_value=np.zeros(2))
        np.testing.assert_array_equal(c.command(None,1.),[0.,0.]);policy.assert_not_called()


if __name__=='__main__':unittest.main()
