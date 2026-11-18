"""Stationary recovery must retain identity and reject stale/moving evidence."""
from pathlib import Path
from unittest.mock import patch
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase_junction_bc import PhasePolicy
from phase_junction_recovery import RecoveringPhaseJunctionController
from phase_junction_controller import PhaseJunctionController
from lidar_patrol_right_then_left import RightThenLeftPatrolController
from junction_localization import JunctionGeometry,JunctionTracker
from navigation import load_config


class WallRecovery(unittest.TestCase):
    def controller(self):
        cfg=load_config(Path(__file__).resolve().parents[2]/'rl_pig_pen/patrol_config_right_then_left.json')
        c=RecoveringPhaseJunctionController(cfg,lambda _:np.zeros(2));c.state='CENTER';c.last_stamp=1.
        g=JunctionGeometry(np.array([-.6,0.]),(0.,np.pi/2),1.,(1.,1.))
        c.tracker=JunctionTracker(g,1.,cfg);c._start_relocalization('junction_geometry_conflict')
        return c,g
    def test_three_current_stationary_measurements_preserve_centre_and_stop(self):
        c,g=self.controller()
        with patch.object(JunctionTracker,'observe_walls',return_value=g):
            for t in (1.2,1.4,1.6):
                c.observe_motion(0.,0.,t)
                np.testing.assert_array_equal(c._relocalization_command(None,t),[0.,0.])
        self.assertIsNone(c.relocalization);self.assertEqual(c.wall_identity_state,'CENTER')
        np.testing.assert_array_equal(c.tracker.geometry.centre,g.centre)
        self.assertEqual(len(c.wall_recovery_events),1)
    def test_parent_actor_cannot_overwrite_zero_handoff(self):
        c,g=self.controller()
        def parent(*args,**kwargs):
            c.wall_recovery_events.append(dict(stamp=1.6))
            c.full_calls+=1;c.full_last_input=np.ones((8,168))
            return np.array([.12,.1])
        with patch.object(PhaseJunctionController,'command',side_effect=parent):
            np.testing.assert_array_equal(c.command(None,1.6),[0.,0.])
        self.assertEqual(c.full_calls,0);self.assertIsNone(c.full_last_input)
        self.assertEqual(c.full_owner,'SAFETY_HOLD')

    def test_moving_or_stale_odometry_never_uses_wall_shortcut(self):
        for v,stamp in ((.03,1.2),(0.,.1)):
            c,g=self.controller();c.observe_motion(v,0.,stamp)
            with patch.object(RightThenLeftPatrolController,'_relocalization_command',return_value=np.zeros(2)) as old:
                c._relocalization_command(None,1.2);old.assert_called_once()
            self.assertEqual(c.wall_recovery_frames,0)
    def test_large_jump_missing_walls_and_long_gap_do_not_resume(self):
        for geometry,t in ((None,1.2),(JunctionGeometry(np.array([-.6,.9]),(0.,np.pi/2),1.,(1.,1.)),1.2),(None,20.)):
            c,g=self.controller();c.observe_motion(0.,0.,t)
            with patch.object(JunctionTracker,'observe_walls',return_value=geometry),patch.object(RightThenLeftPatrolController,'_relocalization_command',return_value=np.zeros(2)):
                c._relocalization_command(None,t)
            self.assertIsNotNone(c.relocalization);self.assertEqual(c.wall_recovery_frames,0)

if __name__=='__main__':unittest.main()
