#!/usr/bin/env python3
"""Evaluate independent full v/w actor against guarded turn-only BC in Gazebo."""
import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import signal
import threading
import time
import numpy as np
from collect_rough_demos import Teacher, transport
from full_junction_bc import FullPolicy,contract,digest,LOCAL_PHASES
from full_junction_controller import FullJunctionController
from rough_demo_common import Case,TASKS,goal_check
from navigation import safe_command
from turn_bc import TurnPolicy


def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def trial(env,case,policy,mode,folder,limit,stop):
    folder.mkdir();env.reset_case(case)
    controller=FullJunctionController(env.cfg,policy) if mode=='full' else Teacher(env.cfg,turn_policy=policy)
    controller.pass_side=case.task.split('_')[0];controller.main_direction=1 if controller.pass_side=='RIGHT' else -1
    controller.state='MAIN' if case.task.endswith('ENTER') else 'REVERSE_LEFT'
    target='LEFT_OUTBOUND' if case.task.endswith('ENTER') else 'MAIN'
    events=[];shifted=False;inhibit_until=None;rows=[];scans=[];inputs=[];input_stamps=[];reason='step_limit'
    hold=blocked_count=0
    with gzip.open(folder/'decisions.jsonl.gz','wt') as log:
        for step in range(limit):
            if stop.is_set():reason='interrupted';break
            if case.turn_disturbance_m and not shifted and controller.state in ('TURN_LEFT','TURN_MAIN'):
                x,y,yaw=env.pose.copy();dx=dy=0.
                if case.disturbance_axis=='incoming_retreat':
                    sign=controller.main_direction if controller.state=='TURN_LEFT' else -1
                    dx=-sign*abs(case.turn_disturbance_m)
                else:dy=case.turn_disturbance_m
                env._publish((0.,0.));env._drain()
                env.teleport(x+math.cos(yaw)*dx-math.sin(yaw)*dy,y+math.sin(yaw)*dx+math.cos(yaw)*dy,yaw,env.pose_z)
                env._drain();env._advance(.2);shifted=True
                if mode=='full':controller.commit([0.,0.])
                events.append(dict(step=step,stamp=env.scan_stamp,type='synthetic_pose_shift',body_offset_m=[dx,dy]))
            if 'actuator_inhibit' in case.family and inhibit_until is None and controller.state in ('TURN_LEFT','TURN_MAIN'):
                duration=12 if 'sustained' in case.family else 4;inhibit_until=step+duration
                events.append(dict(step=step,stamp=env.scan_stamp,type='synthetic_actuator_inhibit',duration_steps=duration))
            features=env._features();before=controller.state
            if mode=='full':
                v=env.odom.twist.twist;controller.observe_motion(v.linear.x,v.angular.z,env.odom_stamp)
            calls=controller.full_calls if mode=='full' else policy.calls
            start=time.perf_counter();command=np.asarray(controller.command(features,env.scan_stamp));latency=(time.perf_counter()-start)*1000
            owned=mode=='full' and controller.full_calls>calls
            turn_owned=mode=='turn' and policy.calls>calls
            executed,blocked=safe_command(command,features,env.cfg,allow_reverse=True)
            inhibited=inhibit_until is not None and step<inhibit_until
            if inhibited:executed=np.zeros(2)
            if owned:
                if before not in LOCAL_PHASES or before!=controller.state:raise RuntimeError('Full BC escaped local ownership')
                inputs.append(controller.full_last_input.copy());input_stamps.append(env.scan_stamp)
            row=dict(step=step,stamp=env.scan_stamp,phase=before,next_phase=controller.state,
                owner='FULL_BC' if owned else 'TURN_BC' if turn_owned else getattr(controller,'full_owner','FSM'),
                command=command.tolist(),executed=executed.tolist(),actuator_inhibited=inhibited,
                model_raw=controller.full_last_raw.tolist() if owned else None,
                teacher_command=controller.full_teacher_command.tolist() if mode=='full' else command.tolist(),
                relocalizing=controller.relocalization is not None,obstacle=controller.obstacle is not None,
                blocked=bool(blocked),retry=controller.turn_reposition_attempts,
                world_pose_privileged=env.pose.tolist(),roll_privileged=env.roll,pitch_privileged=env.pitch,
                controller_ms=latency,diagnostics=controller.diagnostics.copy())
            rows.append(row);scans.append(np.asarray(env.scan.ranges,dtype=np.float32).copy())
            log.write(json.dumps(row,default=default)+'\n');log.flush()
            if step%10==0:env.progress_path.write_text(json.dumps(dict(mode=mode,case=asdict(case),step=step,
                phase=controller.state,owner=row['owner']),indent=2)+'\n')
            if controller.state==target and step>0:
                reason='success' if goal_check(case,env.pose) else 'handoff_world_pose_mismatch';break
            if controller.failure:reason=controller.failure;break
            if abs(env.roll)>.55 or abs(env.pitch)>.65:reason='excessive_tilt';break
            hold=hold+1 if controller.relocalization is not None else 0
            blocked_count=blocked_count+1 if blocked or controller.obstacle is not None else 0
            if hold>50:reason='geometry_hold_timeout';break
            if blocked_count>30:reason='persistent_obstacle';break
            env._publish(executed);env._drain(.02);env._advance(.2)
            if mode=='full':controller.commit(executed)
    env._publish((0.,0.));env._drain();env._advance(.4)
    velocity=env.odom.twist.twist;stationary=abs(velocity.linear.x)<.02 and abs(velocity.angular.z)<.03
    if reason=='success' and not stationary:reason='final_stop_unconfirmed'
    if reason=='success' and mode=='full' and not inputs:reason='no_full_policy_calls'
    np.savez_compressed(folder/'scans.npz',ranges=scans,stamps=[r['stamp'] for r in rows])
    np.savez_compressed(folder/'policy_inputs.npz',history=np.asarray(inputs,dtype=np.float32),stamps=input_stamps)
    result=dict(case=asdict(case),mode=mode,success=reason=='success',reason=reason,steps=len(rows),
        projection_frames=sum(r['diagnostics'].get('full_projection_changed',False) for r in rows),
        full_calls=len(inputs),full_linear_frames=sum(r['owner']=='FULL_BC' and abs(r['executed'][0])>.01 for r in rows),
        full_turn_frames=sum(r['owner']=='FULL_BC' and r['phase'] in ('TURN_LEFT','TURN_MAIN') for r in rows),
        full_centre_frames=sum(r['owner']=='FULL_BC' and r['phase'] in ('CENTER','RETURN_CENTER') for r in rows),
        full_exit_frames=sum(r['owner']=='FULL_BC' and r['phase'] in ('ENTRY_LEFT','EXIT_MAIN') for r in rows),
        max_retries=max((r['retry'] for r in rows),default=0),events=events,
        disturbance_applied=shifted,actuator_inhibit_applied=inhibit_until is not None,
        final_stationary=stationary,final_world_pose=env.pose.tolist(),
        final_measured_velocity=[velocity.linear.x,velocity.angular.z],contacts_instrumented=False,
        safety_frames=sum(r['blocked'] or r['obstacle'] or r['relocalizing'] or r['owner']=='SAFETY_HOLD' or r['diagnostics'].get('full_projection_changed',False) for r in rows))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def planned(stage):
    result=[]
    # Fixed distinct seeds and positions before evaluating any policy; no seed fishing.
    for i,task in enumerate(TASKS):
        c=Case(task,20400000+i,'validation','nominal',(-6.18,0.,6.18)[i%3],.01,math.radians(3),.3)
        result.append(('full',c))
        if stage=='all':result.append(('turn',c))
    if stage=='all':
        for i,task in enumerate(TASKS):
            result.append(('full',Case(task,20400100+i,'test','unseen_incoming',(-6.18,0.,6.18)[(i+1)%3],
                -.01,math.radians(-3),.3,.17,'incoming_retreat')))
        for i,task in enumerate(('RIGHT_ENTER','LEFT_RETURN')):
            result.append(('full',Case(task,20400200+i,'stress','sustained_actuator_inhibit',0.,0.,0.,.2)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--turn-baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('development','all'),default='all');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);policies={'full':FullPolicy(args.model),'turn':TurnPolicy(args.turn_baseline)}
    trials=planned(args.stage);model_files={'full':args.model,'turn':args.turn_baseline}
    meta=dict(contract=contract(),evaluator_sha256=digest(__file__),stage=args.stage,
        models={n:dict(path=str(p.resolve()),sha256=digest(p)) for n,p in model_files.items()},
        plan=[dict(mode=n,case=asdict(c)) for n,c in trials],episodes=[])
    path=args.output/'evaluation.json'
    def persist():path.write_text(json.dumps(meta,indent=2)+'\n')
    persist();stop=threading.Event();env=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        env=transport();env.progress_path=args.output/'progress.json'
        for index,(mode,case) in enumerate(trials):
            if stop.is_set():break
            print('START',index,mode,case.task,case.family,flush=True);folder=args.output/f'{index:02d}_{mode}_{case.task.lower()}'
            try:r=trial(env,case,policies[mode],mode,folder,300,stop)
            except Exception as exc:meta['infrastructure_fault']=repr(exc);persist();raise
            r['folder']=folder.name;meta['episodes'].append(r);persist();print('END',index,r['reason'],flush=True)
    finally:
        if env is not None:env.finish()
        meta['complete']=len(meta['episodes'])==len(trials) and not stop.is_set() and not meta.get('infrastructure_fault')
        meta['models_unchanged']=all(digest(p)==meta['models'][n]['sha256'] for n,p in model_files.items());persist()


if __name__=='__main__':main()
