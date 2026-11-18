"""Phase context is causal; progress recovery and positive feedback are bounded."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase_junction_bc import phase_observation,network,HISTORY,OBS_SIZE,LOCAL_PHASES
from phase_junction_controller import PhaseJunctionController
from navigation import load_config


class PhaseContext(unittest.TestCase):
    def test_centring_target_is_incoming_but_turn_target_is_exit(self):
        raw=np.zeros(159,dtype=np.float32);raw[144]=1;raw[150]=-1;raw[152:154]=1
        centred=phase_observation(raw,'CENTER',(.02,.04))
        turning=phase_observation(raw,'TURN_LEFT',None)
        self.assertEqual(centred.shape,(168,));self.assertAlmostEqual(float(centred[150]),0.,places=6)
        self.assertAlmostEqual(float(centred[151]),1.,places=6)
        self.assertEqual(turning[150],-1);self.assertEqual(centred[159],1)
        np.testing.assert_allclose(centred[165:168],[.02,.04,1.])
        np.testing.assert_array_equal(raw[150:152],[-1,0])
    def test_structured_actor_moves_toward_centre_and_rotates_in_correct_direction(self):
        torch.set_num_threads(1);actor=network()
        for phase in ('CENTER','RETURN_CENTER','TURN_LEFT','TURN_MAIN','ENTRY_LEFT','EXIT_MAIN'):
            raw=np.zeros(159,dtype=np.float32);raw[144]=1;raw[148]=.2;raw[150]=-.2;raw[151]=np.sqrt(.96);raw[152:154]=1
            x=phase_observation(raw,phase,None)
            with torch.inference_mode():y=actor(torch.tensor(np.tile(x,(HISTORY,1))))
            self.assertTrue(torch.isfinite(y).all());self.assertLessEqual(float(y.abs().max()),1.)
            if phase.startswith('TURN'):self.assertEqual(float(y[0]),0.);self.assertLess(float(y[1]),0.)
    def test_actor_gradient_reaches_learned_gain(self):
        actor=network();x=torch.zeros((2,HISTORY,OBS_SIZE));x[:,:,144]=1;x[:,:,150]=-.1;x[:,:,151]=.995;x[:,:,161]=1
        actor(x).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters()),0.)


class ProgressRecovery(unittest.TestCase):
    def controller(self):
        cfg=load_config(Path(__file__).resolve().parents[2]/'rl_pig_pen/patrol_config_right_then_left.json')
        c=PhaseJunctionController(cfg,Mock());c.state='CENTER'
        c.tracker=SimpleNamespace(incoming_axis=0.,geometry=SimpleNamespace(centre=np.array([.21,0.])))
        return c
    def test_zero_command_deadlock_stops_then_uses_counted_fsm_correction(self):
        c=self.controller();self.assertIsNone(c.progress_guard(0.,np.array([.08,0.])))
        np.testing.assert_array_equal(c.progress_guard(2.1,np.array([.08,0.])),[0.,0.])
        self.assertEqual(c.full_owner,'SAFETY_HOLD');self.assertEqual(c.correction_count,1)
        np.testing.assert_array_equal(c.progress_guard(2.3,np.array([.08,0.])),[.08,0.])
        self.assertEqual(c.full_owner,'FSM_CORRECTION')
        c.state='TURN_LEFT';np.testing.assert_array_equal(c.progress_guard(2.5,np.zeros(2)),[0.,0.])
        self.assertIsNone(c.correction_started)
    def test_true_progress_does_not_trigger_and_exhausted_corrections_stop(self):
        c=self.controller();c.progress_guard(0.,np.zeros(2));c.tracker.geometry.centre[0]=.18
        self.assertIsNone(c.progress_guard(1.8,np.zeros(2)));self.assertIsNone(c.progress_guard(3.,np.zeros(2)))
        c.correction_count=2;np.testing.assert_array_equal(c.progress_guard(4.,np.zeros(2)),[0.,0.])
        self.assertEqual(c.failure,'phase_no_progress_limit')
    def test_exit_is_slow_until_consecutive_stable_wall_measurements(self):
        c=self.controller();c.state='ENTRY_LEFT';c.entry_phase='WALL_CONFIRM';c.tracker=None
        c.policy=Mock(return_value=np.array([.2,.2]));c._safe=Mock(side_effect=lambda action,f:np.asarray(action))
        raw=np.zeros(159,dtype=np.float32);raw[144]=1.;raw[151]=1.;raw[153]=1.
        outputs=[]
        with patch('phase_junction_controller.RightThenLeftPatrolController.command',return_value=np.zeros(2)), \
                patch('phase_junction_controller.incoming_corridor',return_value=(.02,.04)), \
                patch('phase_junction_controller.observation',return_value=raw):
            for stamp in (1.,1.2,1.4):
                c.observe_motion(.05,0.,stamp);outputs.append(c.command(None,stamp)[0]);c.commit([outputs[-1],0.])
        np.testing.assert_allclose(outputs,[.07,.07,.12])

    def test_correction_has_fixed_timeout(self):
        c=self.controller();c.correction_started=0.;c.correction_phase='CENTER'
        np.testing.assert_array_equal(c.progress_guard(8.1,np.array([.08,0.])),[0.,0.])
        self.assertEqual(c.failure,'phase_correction_timeout')


if __name__=='__main__':unittest.main()
