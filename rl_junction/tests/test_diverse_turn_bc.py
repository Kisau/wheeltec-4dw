"""Only valid causal pure-turn labels may enter the guarded turn dataset."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_diverse_turn_bc import turn_mask


class TurnExtraction(unittest.TestCase):
    def test_mask_rejects_transition_invalid_aligned_moving_and_wrong_direction(self):
        x=np.zeros((7,159));x[:,150]=np.sin(.2);x[:,151]=np.cos(.2)
        a=dict(observations=x,label_valid=np.ones(7,dtype=bool),
            phases=np.array(['TURN_LEFT']*7),next_phases=np.array(['TURN_LEFT']*7),
            teacher_actions=np.tile([0.,.4],(7,1)))
        a['next_phases'][1]='ENTRY_LEFT';a['label_valid'][2]=False
        x[3,150]=0.;x[3,151]=1.;a['teacher_actions'][4,0]=.1
        a['teacher_actions'][5,1]=-.4;a['phases'][6]='CENTER'
        np.testing.assert_array_equal(turn_mask(a,.06),[True,False,False,False,False,False,False])

    def test_return_turn_negative_heading_is_valid(self):
        x=np.zeros((1,159));x[0,150]=np.sin(-.5);x[0,151]=np.cos(-.5)
        a=dict(observations=x,label_valid=np.ones(1,dtype=bool),
            phases=np.array(['TURN_MAIN']),next_phases=np.array(['TURN_MAIN']),
            teacher_actions=np.array([[0.,-.5]]))
        self.assertTrue(turn_mask(a,.06)[0])


if __name__=='__main__':unittest.main()
