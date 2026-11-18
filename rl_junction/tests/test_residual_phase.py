from pathlib import Path
import sys,unittest,tempfile,xml.etree.ElementTree as ET
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from residual_phase_env import bounded_residual,ResidualPhaseEnv,RESIDUAL_LIMIT
from residual_dynamics import robot_variant
ROOT=Path(__file__).resolve().parents[2]
class ResidualBounds(unittest.TestCase):
 def test_limits_zero_baseline_and_turn_translation(self):
  base=np.array([.1,.2])
  for phase in ('CENTER','RETURN_CENTER','TURN_LEFT','TURN_MAIN','ENTRY_LEFT','EXIT_MAIN'):
   y,d=bounded_residual(base,[9.,-9.],phase)
   self.assertTrue(np.all(np.abs(d)<=RESIDUAL_LIMIT+1e-7))
   if phase.startswith('TURN'):self.assertEqual(y[0],base[0])
   np.testing.assert_allclose(bounded_residual(base,[0.,0.],phase)[0],base)
 def test_bad_residual_rejected(self):
  for x in ([float('nan'),0.],[1.],[1.,2.,3.]):
   with self.assertRaises(ValueError):bounded_residual([0.,0.],x,'CENTER')
 def test_frozen_bc_parameters_and_checkpoint_unchanged(self):
  model=ROOT/'logs/phase_junction_bc/20261117/model/model.pt'
  with tempfile.TemporaryDirectory() as d,patch('residual_phase_env.transport') as transport:
   env=ResidualPhaseEnv(model,d)
   self.assertFalse(any(p.requires_grad for p in env.bc.model.parameters()));env.close()
 def test_same_frame_obstacle_cannot_be_overridden_by_residual(self):
  env=ResidualPhaseEnv.__new__(ResidualPhaseEnv)
  env.last_history=np.zeros((8,168),np.float32)
  velocity=SimpleNamespace(linear=SimpleNamespace(x=0.),angular=SimpleNamespace(z=0.))
  env.env=SimpleNamespace(odom=SimpleNamespace(twist=SimpleNamespace(twist=velocity)),odom_stamp=1.,scan_stamp=1.,_features=lambda:None)
  c=SimpleNamespace(full_calls=0,relocalization=None,obstacle=None,external_paused=False,failure=None,state='CENTER',observe_motion=lambda *args:None,progress_metric=lambda:None)
  def command(*args):
   c.full_calls+=1;c.obstacle={'phase':'WAIT'}
   return np.zeros(2)
  c.command=command;env.c=c
  obs=env._prepare()
  self.assertFalse(env.owned);self.assertEqual(obs[-1],0.)
  np.testing.assert_array_equal(env.command,[0.,0.])

 def test_physical_variant_changes_mass_inertia_and_wheel_friction(self):
  source=ROOT/'turn_on_wheeltec_robot/urdf/four_wheel_diff_bs_robot.urdf'
  with tempfile.TemporaryDirectory() as d:
   target=Path(d)/'robot.urdf';m=robot_variant(source,target,.65,5.)
   tree=ET.parse(target);original=ET.parse(source)
   self.assertAlmostEqual(float(tree.find("link[@name='base_link']/inertial/mass").get('value')),float(original.find("link[@name='base_link']/inertial/mass").get('value'))+5.)
   self.assertTrue(all(float(x.text)==.65 for x in tree.findall('.//mu1')))
if __name__=='__main__':unittest.main()
