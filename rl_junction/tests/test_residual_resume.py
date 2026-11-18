"""Offline checks for optimizer continuation and TensorBoard; no Gazebo involved."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_residual_phase as trainer

class TinyEnv(gym.Env):
 def __init__(self,bc,output,seed=0):
  self.bc_hash=trainer.digest(bc);self.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32)
  self.observation_space=gym.spaces.Box(-1.,1.,(8,),np.float32);self.n=0
 def reset(self,seed=None,options=None):super().reset(seed=seed);self.n=0;return np.zeros(8,np.float32),{}
 def step(self,action):
  self.n+=1;return np.zeros(8,np.float32),-float(np.sum(np.square(action))),False,self.n>=16,{}
 def close(self):pass

class ResumeChecks(unittest.TestCase):
 def test_resume_keeps_parameters_optimizer_and_step_counter(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);bc=root/'bc.pt';bc.write_bytes(b'frozen-bc-test')
   model=PPO('MlpPolicy',TinyEnv(bc,root),n_steps=64,batch_size=32,n_epochs=1,policy_kwargs=dict(net_arch=[16]),verbose=0)
   model.learn(64);model.save(root/'residual_ppo')
   (root/'training.json').write_text(json.dumps(dict(steps=64,complete=True,seed=0,bc_sha256=trainer.digest(bc),residual_limits=trainer.RESIDUAL_LIMIT.tolist())))
   expected=model.policy.action_net.weight.detach().clone();old_learn=PPO.learn;seen=[]
   def learning(loaded,*args,**kwargs):
    self.assertEqual(loaded.num_timesteps,64)
    self.assertTrue(torch.equal(loaded.policy.action_net.weight,expected))
    self.assertTrue(loaded.policy.optimizer.state_dict()['state'])
    self.assertFalse(kwargs['reset_num_timesteps']);seen.append(True)
    return old_learn(loaded,*args,**kwargs)
   argv=['train','--bc',str(bc),'--resume',str(root/'residual_ppo.zip'),'--output',str(root/'continued'),'--steps','64']
   with patch.object(sys,'argv',argv),patch.object(trainer,'ResidualPhaseEnv',TinyEnv),patch.object(PPO,'learn',learning):trainer.main()
   meta=json.loads((root/'continued/training.json').read_text())
   self.assertTrue(seen);self.assertEqual(meta['starting_steps'],64);self.assertEqual(meta['steps'],128)
   events=EventAccumulator(str(root/'continued/metrics'));events.Reload()
   self.assertEqual(events.Scalars('train/value_loss')[-1].step,128)
 def test_wrong_frozen_bc_rejected_before_training(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);bc=root/'bc.pt';bc.write_bytes(b'wrong');checkpoint=root/'residual_ppo.zip';checkpoint.write_bytes(b'test')
   (root/'training.json').write_text(json.dumps(dict(complete=True,bc_sha256='wrong-hash',residual_limits=[.03,.12])))
   with self.assertRaisesRegex(ValueError,'same frozen BC'):trainer.validate_parent(checkpoint,bc)

if __name__=='__main__':unittest.main()
