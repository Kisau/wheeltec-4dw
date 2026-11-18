"""Causal history/split supervision and full-v/w runtime safety ownership."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_junction_bc import history_windows,eligible_mask,HISTORY
from navigation import load_config
from full_junction_controller import FullJunctionController


class HistoryTests(unittest.TestCase):
    def test_history_does_not_use_future_and_does_not_cross_episodes(self):
        x=np.arange(5)[:,None].astype(np.float32);windows=history_windows(x,3)
        np.testing.assert_array_equal(windows[:,:,0],[[0,0,0],[0,0,1],[0,1,2],[1,2,3],[2,3,4]])
        np.testing.assert_array_equal(history_windows(np.array([[99.]]),3),[[[99.],[99.],[99.]]])
    def test_supervision_excludes_straight_transition_invalid(self):
        data=dict(label_valid=np.array([1,1,0,1],dtype=bool),
            phases=np.array(['CENTER','MAIN','TURN_LEFT','TURN_MAIN']),
            next_phases=np.array(['CENTER','MAIN','TURN_LEFT','EXIT_MAIN']))
        np.testing.assert_array_equal(eligible_mask(data),[True,False,False,False])


class FullOwnership(unittest.TestCase):
    def fixture(self):
        cfg=load_config(Path(__file__).resolve().parents[2]/'rl_pig_pen/patrol_config_right_then_left.json')
        policy=Mock(return_value=np.array([.08,-.3]));c=FullJunctionController(cfg,policy)
        c.state='CENTER';c.tracker=SimpleNamespace(stamp=1.,incoming_axis=-.1,geometry=SimpleNamespace(centre=np.array([.5*np.cos(-.1),.5*np.sin(-.1)])))
        c.observe_motion(.04,0.,1.);c._safe=Mock(side_effect=lambda action,f:np.asarray(action))
        obs=np.zeros(159,dtype=np.float32);obs[153]=1.;return c,policy,obs
    def invoke(self,c,obs):
        with patch('full_junction_controller.RightThenLeftPatrolController.command',return_value=np.zeros(2)), \
                patch('full_junction_controller.incoming_corridor',return_value=None), \
                patch('full_junction_controller.observation',return_value=obs):
            return c.command(None,1.)
    def test_actor_can_own_linear_and_angular_before_turn(self):
        c,p,x=self.fixture();np.testing.assert_allclose(self.invoke(c,x),[.08,-.3])
        self.assertEqual(c.full_owner,'FULL_BC');self.assertEqual(p.call_args[0][0].shape,(HISTORY,159))
    def test_straight_external_stop_stale_geometry_do_not_call_actor(self):
        for setting in ('straight','external','odom','geometry'):
            with self.subTest(setting=setting):
                c,p,x=self.fixture()
                if setting=='straight':c.state='MAIN'
                if setting=='external':c.external_paused=True
                if setting=='odom':c.measured_stamp=0.
                if setting=='geometry':x[153]=0.
                np.testing.assert_array_equal(self.invoke(c,x),[0.,0.]);p.assert_not_called()
    def test_invalid_actor_fails_closed_and_retry_caps_angular(self):
        c,p,x=self.fixture();p.return_value=np.array([np.nan,0.])
        np.testing.assert_array_equal(self.invoke(c,x),[0.,0.]);self.assertEqual(c.failure,'full_policy_fault')
        c,p,x=self.fixture();c.turn_reposition_attempts=1;p.return_value=np.array([2.,-2.])
        np.testing.assert_allclose(self.invoke(c,x),[.05,-.25])
    def test_geometric_projection_prevents_early_turn_and_holds_alignment(self):
        c,p,x=self.fixture();c.tracker.incoming_axis=0.;c.tracker.geometry.centre[:]=[.5,0.]
        np.testing.assert_allclose(c.project_local([.1,-.5]),[.1,0.])
        c.tracker.geometry.centre[:]=[.02,0.]
        np.testing.assert_allclose(c.project_local([.1,-.5]),[0.,0.])
        c.state='TURN_LEFT';c.tracker.incoming_axis=np.pi/2
        np.testing.assert_allclose(c.project_local([.1,-.5]),[0.,0.])

    def test_nonmotion_watchdog_stops_before_another_actor_call(self):
        c,p,x=self.fixture();c.measured[:]=0.;c.last_requested[:]=[0.,.3];c.no_motion_frames=7
        np.testing.assert_array_equal(self.invoke(c,x),[0.,0.]);p.assert_not_called()
        self.assertEqual(c.failure,'full_no_motion_watchdog')


if __name__=='__main__':unittest.main()
