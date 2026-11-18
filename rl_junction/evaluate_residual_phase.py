#!/usr/bin/env python3
"""Paired frozen BC / residual PPO on fresh physical initial/dynamics conditions."""
import argparse,json,math,shutil,signal,threading,time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from residual_phase_env import ResidualPhaseEnv
from residual_dynamics import verify_world_variant
from residual_scene import ExperimentScene
from phase_junction_bc import digest
from rough_demo_common import Case,TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--bc',type=Path,required=True);p.add_argument('--ppo',type=Path,required=True);p.add_argument('--scene-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=a.resume);manager=ExperimentScene(a.scene_root)
 training=json.loads(a.ppo.with_name('training.json').read_text())
 if digest(a.bc)!=training['bc_sha256']:raise ValueError('Residual checkpoint paired with wrong BC')
 model=PPO.load(a.ppo,device='cpu');env=None;entries=[]
 meta=dict(sources={f:digest(Path(__file__).parent/f) for f in ('residual_phase_env.py','residual_dynamics.py','residual_scene.py','evaluate_residual_phase.py')},bc_sha256=digest(a.bc),ppo_sha256=digest(a.ppo),plan=[],episodes=[],complete=False,restored_original=False)
 path=a.output/'evaluation.json'
 def save():path.write_text(json.dumps(meta,indent=2)+'\n')
 scenarios=[('offset',.08,-3.,1.,0.),('yaw',-.03,8.,1.,0.),('low_friction',.04,-4.,.65,0.),('payload',-.04,4.,1.,5.)]
 for i,(name,lat,yaw,mu,kg) in enumerate(scenarios):
  case=Case(TASKS[i],21100000+i,'test',name,(-6.18,0.,6.18)[i%3],lat,math.radians(yaw),.3)
  meta['plan'].append(dict(case=case.__dict__,friction=mu,payload_kg=kg))
 if a.resume:
  old=json.loads(path.read_text())
  if any(old[k]!=meta[k] for k in ('bc_sha256','ppo_sha256','plan','sources')):raise ValueError('Resume model/plan/source mismatch')
  meta=old;meta['complete']=False;meta['restored_original']=False
 snapshot=a.output/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in meta['sources']:shutil.copy2(Path(__file__).parent/name,snapshot/name)
 save();stop=threading.Event()
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())

 try:
  for i,scenario in enumerate(meta['plan']):
   for mode in ('bc','residual'):
    if any(e['scenario']==i and e['mode']==mode for e in meta['episodes']):continue
    if stop.is_set():break
    # Respawn BOTH members of a pair, including nominal physics, to equalize
    # solver state and ensure the requested physical parameters were applied.
    if env is not None:env.close();env=None
    variant=manager.start(f'{a.output.name}_{i}_{mode}_{time.time_ns()}',scenario['friction'],scenario['payload_kg'])
    env=ResidualPhaseEnv(a.bc,a.output/f'rollout_{i}_{mode}')
    variant['live_verification']=verify_world_variant(variant,a.output/f'live_{i}_{mode}.sdf')
    env.log.close();env.log=(a.output/f'transitions_{i}_{mode}.jsonl').open('w',buffering=1)
    env.cases=[Case(**scenario['case'])];env.episode=0
    obs,_=env.reset();steps=[];terminated=truncated=False
    print('START',i,mode,scenario['case']['task'],flush=True)
    while not (terminated or truncated) and not stop.is_set():
     action=np.zeros(2) if mode=='bc' else model.predict(obs,deterministic=True)[0]
     obs,reward,terminated,truncated,info=env.step(action);steps.append(info)
    if stop.is_set():break
    row=dict(scenario=i,mode=mode,case=scenario['case'],dynamics=variant,success=info['success'],reason=info['reason'],final_stationary=info['final_stationary'],final_world_pose=info['world_pose_privileged'],steps=len(steps),fsm_correction_frames=sum(x['fsm_correction'] for x in steps),projection_frames=sum(x['projection'] for x in steps),obstacle_frames=sum(x['blocked'] for x in steps),wall_recoveries=env.c.wall_recovery_events,
     exit_offset_m=abs(info['world_pose_privileged'][1]-scenario['case']['junction_y']) if scenario['case']['task'].endswith('ENTER') else abs(info['world_pose_privileged'][0]))
    meta['episodes'].append(row);save();print('END',i,mode,row['success'],row['reason'],flush=True)
  meta['complete']=len(meta['episodes'])==8 and not stop.is_set()
 finally:
  try:
   if env is not None:env.close();env=None
   manager.start(f'{a.output.name}_restored_nominal_{time.time_ns()}');meta['restored_original']=True
  finally:save()
if __name__=='__main__':main()
