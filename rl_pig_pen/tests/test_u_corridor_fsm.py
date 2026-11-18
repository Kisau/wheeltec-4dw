import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from u_corridor_fsm import UCorridorFSM


class UCorridorTests(unittest.TestCase):
    clear=[(2*math.cos(i*.03),2*math.sin(i*.03)) for i in range(210)]

    def test_complete_forward_reverse_route(self):
        c=UCorridorFSM(); pose=[0.,0.,0.]; measured=(0.,0.)
        seen=[]
        for i in range(20000):
            index=c.index
            v,w=c.step(tuple(pose),measured,self.clear,i*.05)
            self.assertNotEqual(c.state,'FAILED',c.status())
            if v: self.assertEqual(v>0,index<3)
            if c.state=='ALIGN': self.assertEqual(v,0)
            pose[0]+=v*math.cos(pose[2])*.05
            pose[1]+=v*math.sin(pose[2])*.05
            pose[2]+=w*.05; measured=(v,w)
            if c.state=='DONE': break
        self.assertEqual(c.state,'DONE')
        self.assertEqual(c.visits,['A','B','C','D','C','B','A'])
        self.assertLess(math.hypot(*pose[:2]),.06)
        self.assertEqual(c.step(tuple(pose),(0,0),self.clear,10000),(0,0))

    def test_reverse_obstacle(self):
        c=UCorridorFSM();c.index=3;c.state='DRIVE'
        cmd=c.step((0.,-22.,-math.pi),(0,0),self.clear+[(-.4,0)],0)
        self.assertEqual(cmd,(0,0));self.assertEqual(c.reason,'TRAVEL_BLOCKED')

    def test_rotation_blocked(self):
        c=UCorridorFSM();c.index=1
        self.assertEqual(c.step((23.,0.,0.),(0,0),self.clear+[(0,.4)],0),(0,0))
        self.assertEqual(c.reason,'ROTATION_BLOCKED')

    def test_no_progress(self):
        c=UCorridorFSM();c.state='DRIVE'
        c.step((0,0,0),(0,0),self.clear,0)
        self.assertEqual(c.step((0,0,0),(0,0),self.clear,3.1),(0,0))
        self.assertEqual(c.reason,'NO_PROGRESS')


if __name__=='__main__':unittest.main()
