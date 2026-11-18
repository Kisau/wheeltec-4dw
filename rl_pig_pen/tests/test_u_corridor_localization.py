import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from u_corridor_localization import WallLocalizer
from u_corridor_fsm import wrap


class WallLocalizationTests(unittest.TestCase):
    def make(self):
        return WallLocalizer(Path(__file__).resolve().parents[2]/'worlds/u_corridor_flat.world',
                             (0,-11.3,math.pi/2))

    def points(self,loc,pose):
        out=[]
        for i in range(1640):
            angle=-math.pi+i*2*math.pi/1639
            direction=(math.cos(angle+pose[2]),math.sin(angle+pose[2]))
            best=25.
            for axis,value,lo,hi in loc.segments:
                if abs(direction[axis])<1e-8:continue
                t=(value-pose[axis])/direction[axis]
                if .1<t<best and lo<=pose[1-axis]+t*direction[1-axis]<=hi:best=t
            if best<25:
                noise=.008*math.sin(i*2.13)
                out.append(((best+noise)*math.cos(angle),(best+noise)*math.sin(angle)))
        return out

    def test_wall_matching_recovers_turn_slip_and_translation(self):
        for actual in ((0,0,0),(23,0,-.7),(23,-10,-math.pi/2),
                       (23,-22,-2.5),(0,-22,-math.pi)):
            with self.subTest(actual=actual):
                loc=self.make();loc.predict((actual[0]+.05,actual[1]-.04,actual[2]+.06))
                self.assertTrue(loc.correct(self.points(loc,actual)),loc.quality)
                self.assertLess(math.hypot(loc.pose[0]-actual[0],loc.pose[1]-actual[1]),.025)
                self.assertLess(abs(wrap(loc.pose[2]-actual[2])),.006)

    def test_missing_map_observations_rejected(self):
        loc=self.make();loc.predict((0,0,0));self.assertFalse(loc.correct([]))


if __name__=='__main__':unittest.main()
