"""Check four-task coverage, split grouping and privileged-pose scoring."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rough_demo_common import Case, TASKS, plan, observation, goal_check, ROOT
from navigation import load_config


class RoughDemoContract(unittest.TestCase):
    def test_plan_has_unique_episodes_and_all_tasks_in_each_split(self):
        cases=plan()
        self.assertEqual(len(cases),32)
        self.assertEqual(len({c.seed for c in cases}),32)
        for split in ('train','validation','test'):
            self.assertEqual({c.task for c in cases if c.split==split},set(TASKS))
        self.assertTrue(all(c.split=='train' for c in cases if c.family=='nominal'))
        self.assertEqual({c.cruise_m_s for c in cases},{.2,.3,.4})
        self.assertTrue(any(c.turn_disturbance_m>0 for c in cases))
        self.assertEqual({c.turn_disturbance_m for c in cases if c.turn_disturbance_m},{.16,.18})
        self.assertTrue(all(c.disturbance_axis=='incoming_retreat' for c in cases if c.turn_disturbance_m))

    def test_spawn_respects_raised_main_and_lower_branch(self):
        for c in plan():
            x,y,yaw,z=c.start()
            if c.task.endswith('ENTER'):
                self.assertEqual(z,.11)
                self.assertLess(abs(x),.1)
                self.assertGreater((c.junction_y-y)*(1 if c.task.startswith('RIGHT') else -1),1.)
            else:
                self.assertEqual(z,.05)
                self.assertAlmostEqual(abs(x),1.2)

    def test_independent_goal_rejects_wrong_side_and_wrong_heading(self):
        poses={'RIGHT_ENTER':(.9,0,0),'LEFT_ENTER':(-.9,0,math.pi),
               'RIGHT_RETURN':(0,.9,math.pi/2),'LEFT_RETURN':(0,-.9,math.pi/2)}
        for task,pose in poses.items():
            c=Case(task,1,'train','test',0,0,0,.2)
            self.assertTrue(goal_check(c,pose))
            self.assertFalse(goal_check(c,(-pose[0],-pose[1],pose[2])))
            self.assertFalse(goal_check(c,(pose[0],pose[1],pose[2]+math.pi/2)))

    def test_observation_has_masks_and_no_world_pose_or_phase(self):
        cfg=load_config(ROOT/'rl_pig_pen/patrol_config_right_then_left.json')
        f=SimpleNamespace(ranges=np.ones(1640),angles=np.linspace(-math.pi,math.pi,1640))
        x=observation(f,cfg,'RIGHT_ENTER',None,(.1,0),(.2,.3),(0,0),.2)
        self.assertEqual(x.shape,(159,))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[152],0.)
        self.assertEqual(x[153],1.)
        self.assertTrue(np.any(x[72:144]==0.))


if __name__=='__main__':unittest.main()
