"""Physical Gazebo residual PPO; frozen BC and external supervisors retained."""
from pathlib import Path
from dataclasses import asdict
import json,math
import numpy as np
import gymnasium as gym
from phase_junction_bc import PhasePolicy,HISTORY,OBS_SIZE,digest
from phase_junction_recovery import RecoveringPhaseJunctionController
from collect_rough_demos import transport
from rough_demo_common import Case,TASKS,goal_check
from navigation import safe_command

RESIDUAL_LIMIT=np.array([.03,.12],dtype=np.float32)

def bounded_residual(base, action, phase):
 action=np.asarray(action,dtype=np.float32)
 if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Invalid residual')
 delta=np.clip(action,-1.,1.)*RESIDUAL_LIMIT
 # Preserve the geometric pure-rotation contract.
 if phase in ('TURN_LEFT','TURN_MAIN'):delta[0]=0.
 return np.asarray(base)+delta,delta

class ResidualPhaseEnv(gym.Env):
 metadata={}
 def __init__(self,bc_path,output,seed=0,cases=None):
  self.bc=PhasePolicy(bc_path);self.bc_path=Path(bc_path);self.bc_hash=digest(bc_path)
  for p in self.bc.model.parameters():p.requires_grad_(False)
  self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
  self.log=(self.output/'transitions.jsonl').open('w',buffering=1)
  self.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32)
  self.observation_space=gym.spaces.Box(-np.inf,np.inf,(HISTORY*OBS_SIZE+1,),np.float32)
  self.env=transport();self.env.progress_path=self.output/'progress.json'
  self.rng=np.random.default_rng(seed);self.episode=0;self.steps=0;self.last_history=np.zeros((HISTORY,OBS_SIZE),np.float32)
  self.cases=cases;self.active=False
 def _prepare(self):
  e=self.env;v=e.odom.twist.twist;self.c.observe_motion(v.linear.x,v.angular.z,e.odom_stamp)
  self.features=e._features();calls=self.c.full_calls
  self.command=np.asarray(self.c.command(self.features,e.scan_stamp));self.owned=(self.c.full_calls>calls and self.c.relocalization is None
   and self.c.obstacle is None and not self.c.external_paused and not self.c.failure)
  if self.owned:self.last_history=self.c.full_last_input.copy()
  self.metric=self.c.progress_metric();self.phase=self.c.state
  return np.r_[self.last_history.ravel(),float(self.owned)].astype(np.float32)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed)
  if self.cases:case=self.cases[self.episode%len(self.cases)]
  else:
   i=self.episode%4;case=Case(TASKS[i],21000000+self.episode,'train','residual_initial_offset',(-6.18,0.,6.18)[self.episode%3],float(self.rng.uniform(-.06,.06)),float(self.rng.uniform(-.1,.1)),.3)
  self.episode+=1;self.case=case;self.env.reset_case(case)
  self.c=RecoveringPhaseJunctionController(self.env.cfg,self.bc);self.c.pass_side=case.task.split('_')[0];self.c.main_direction=1 if self.c.pass_side=='RIGHT' else -1
  self.c.state='MAIN' if case.task.endswith('ENTER') else 'REVERSE_LEFT'
  self.target='LEFT_OUTBOUND' if case.task.endswith('ENTER') else 'MAIN'
  self.steps=0;self.hold=0;self.previous_delta=np.zeros(2);self.active=True
  self.last_history=np.zeros((HISTORY,OBS_SIZE),np.float32)
  return self._prepare(),dict(case=asdict(case))
 def step(self,action):
  if not self.active:raise RuntimeError('Reset required')
  owner_before=self.c.full_owner
  before=self.phase;metric_before=self.metric;owned=self.owned;delta=np.zeros(2);command=self.command.copy();projected=False
  if owned:
   base=self.bc(self.last_history);request,delta=bounded_residual(base,action,before)
   angular=min(self.c.cfg.turn_speed,.25) if self.c.turn_reposition_attempts else self.c.cfg.turn_speed
   bounded=np.clip(request,[-min(.4,self.c.cfg.cruise_speed),-angular],[min(.4,self.c.cfg.cruise_speed),angular])
   command=self.c.project_local(bounded)
   if before in self.c.ENTRY_STATES:command[0]=np.clip(command[0],-.12 if self.c.exit_stable_frames>=self.c.cfg.confirm_frames else -.07,.12 if self.c.exit_stable_frames>=self.c.cfg.confirm_frames else .07)
   projected=not np.allclose(command,request,atol=1e-7)
   self.c.last_requested=command.copy()
  executed,blocked=safe_command(command,self.features,self.c.cfg,allow_reverse=True)
  self.env._publish(executed);self.env._drain(.02);self.env._advance(.2);self.c.commit(executed)
  self.steps+=1;obs=self._prepare();self.hold=self.hold+1 if self.c.relocalization is not None else 0
  success=self.c.state==self.target and goal_check(self.case,self.env.pose)
  reason='success' if success else self.c.failure or ('geometry_hold_timeout' if self.hold>50 else None)
  if abs(self.env.roll)>.55 or abs(self.env.pitch)>.65:reason='excessive_tilt';success=False
  terminated=reason is not None;truncated=self.steps>=250 and not terminated
  valid=owned and not blocked and self.c.relocalization is None and self.c.obstacle is None
  progress=0.
  if valid and self.phase==before and metric_before is not None and self.metric is not None:
   gain=2. if before.startswith('TURN') else 8.;progress=gain*float(np.clip(metric_before[0]-self.metric[0],-.08,.08))
  # Safety waits are neither punished nor rewarded. Progress is geometric,
  # never just speed; stopping cannot earn terminal success.
  components=dict(progress=progress,time=-.005 if valid else 0.,residual=-.002*float(delta@delta),
   smooth=-.001*float(np.sum((delta-self.previous_delta)**2)),projection=-.02 if valid and projected else 0.,
   terminal=20. if success else -10. if terminated else 0.)
  self.previous_delta=delta.copy()
  info=dict(success=bool(success),reason=reason or ('time_limit' if truncated else 'running'),owner=owner_before,
   phase=before,next_phase=self.phase,base_owned=owned,residual=delta.tolist(),executed=executed.tolist(),blocked=bool(blocked),projection=projected,
   fsm_correction=owner_before=='FSM_CORRECTION',reward_components=components)
  if terminated or truncated:
   self.env._publish((0.,0.));self.env._drain();self.env._advance(.4)
   v=self.env.odom.twist.twist;info['final_stationary']=abs(v.linear.x)<.02 and abs(v.angular.z)<.03
   info['world_pose_privileged']=self.env.pose.tolist();self.active=False
   if success and not info['final_stationary']:
    info['success']=False;info['reason']='final_stop_unconfirmed';components['terminal']=-10.
  self.log.write(json.dumps(dict(episode=self.episode,step=self.steps,case=asdict(self.case),**info))+'\n')
  return obs,sum(components.values()),terminated,truncated,info
 def close(self):
  self.env.finish();self.log.close()
  if digest(self.bc_path)!=self.bc_hash:raise RuntimeError('Frozen BC changed')
