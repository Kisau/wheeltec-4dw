#!/usr/bin/env python3
"""Additional teacher correction trajectories and explicit non-motion failures.

Separate from the old collector: original manifests and model contracts unchanged.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import threading
import numpy as np
from collect_rough_demos import Teacher, transport, save_episode
from rough_demo_common import ROOT, TASKS, Case, contract, observation, goal_check
from navigation import safe_command
from lidar_patrol_right_then_left import incoming_corridor


def correction_episode(env,case,limit,stop):
    env.current_case_frames=[]
    env.reset_case(case)
    teacher=Teacher(env.cfg);teacher.pass_side=case.task.split('_')[0]
    teacher.main_direction=1 if teacher.pass_side=='RIGHT' else -1
    teacher.state='MAIN' if case.task.endswith('ENTER') else 'REVERSE_LEFT'
    target='LEFT_OUTBOUND' if case.task.endswith('ENTER') else 'MAIN'
    previous=np.zeros(2);frames=[];events=[];disturbed=False;hold_steps=0;blocked_steps=0;reason='step_limit'
    env.current_case_frames=frames
    inhibit_until=None; no_motion=0; last_probe=None
    for step in range(limit):
        if stop.is_set():reason='interrupted';break
        if case.turn_disturbance_m and not disturbed and teacher.state in ('TURN_LEFT','TURN_MAIN'):
            x,y,yaw=env.pose.copy();dx=dy=0.
            if case.disturbance_axis=='incoming_retreat':
                # Move back along the approach, where the existing teacher can
                # safely re-centre. Large lateral shifts exceed its hard bound.
                approach=teacher.main_direction if teacher.state=='TURN_LEFT' else -1
                dx=-approach*abs(case.turn_disturbance_m)
            else:dy=case.turn_disturbance_m
            env._publish((0.,0.));env._drain();env.teleport(
                x+math.cos(yaw)*dx-math.sin(yaw)*dy,
                y+math.sin(yaw)*dx+math.cos(yaw)*dy,yaw,env.pose_z)
            env._drain();env._advance(.2);previous[:]=0.;disturbed=True
            events.append(dict(step=step,type='synthetic_turn_pose_shift',body_offset_m=[dx,dy],
                axis=case.disturbance_axis,
                stamp=env.scan_stamp,note='Discrete pose perturbation; not a physical slip/contact model'))
        if 'actuator_inhibit' in case.family and inhibit_until is None and teacher.state in ('TURN_LEFT','TURN_MAIN'):
            duration=12 if 'sustained' in case.family else 4
            inhibit_until=step+duration
            events.append(dict(step=step,type='synthetic_actuator_inhibit',duration_steps=duration,
                stamp=env.scan_stamp,note='Published velocity suppressed in software, not physical wheel slip'))
        features=env._features();before=teacher.state;teacher.last_safety=None
        command=np.asarray(teacher.command(features,env.scan_stamp),dtype=np.float32)
        corridor=incoming_corridor(features,env.cfg)
        velocity=env.odom.twist.twist
        fresh_tracker=(teacher.tracker if teacher.tracker is not None and
                       abs(teacher.tracker.stamp-env.scan_stamp)<1e-5 else None)
        exit_corridor=(corridor if teacher.state==target or
            (teacher.state in teacher.ENTRY_STATES and teacher.entry_phase=='WALL_CONFIRM') else None)
        measured_dt=env.scan_stamp-frames[-1]['stamp'] if frames else .2
        x=observation(features,env.cfg,case.task,fresh_tracker,exit_corridor,
            (velocity.linear.x,velocity.angular.z),previous,measured_dt)
        executed,blocked=safe_command(command,features,env.cfg,allow_reverse=True)
        inhibited=inhibit_until is not None and step<inhibit_until
        if inhibited:executed=np.zeros(2,dtype=np.float32)
        measured_speed=abs(velocity.linear.x)+.2*abs(velocity.angular.z)
        demanded=last_probe is not None and (abs(last_probe[0])>.025 or abs(last_probe[1])>.1)
        no_motion=no_motion+1 if demanded and measured_speed<.008 else 0
        stalled=no_motion>=8
        handoff=teacher.state==target and step>0
        valid=bool(not teacher.failure and teacher.relocalization is None and teacher.obstacle is None
                   and not teacher.external_paused and not blocked and not handoff
                   and x[153]>.5 and before==teacher.state and not inhibited and not stalled)
        requested=command.copy();request_available=False
        if teacher.last_safety is not None:
            raw,safe=teacher.last_safety
            if before==teacher.state and np.allclose(safe,command):requested=raw;request_available=True
        frames.append(dict(observation=x,teacher_action=command/[.4,.6],executed_action=executed/[.4,.6],
            safety_request=requested,request_available=request_available,label_valid=valid,
            stamp=env.scan_stamp,clock=env.sim_time,phase=before,next_phase=teacher.state,world_pose=env.pose.copy(),
            z=env.pose_z,roll=env.roll,pitch=env.pitch,raw=np.asarray(env.scan.ranges,dtype=np.float32).copy(),
            relocalizing=teacher.relocalization is not None,obstacle=teacher.obstacle is not None,
            retry=teacher.turn_reposition_attempts))
        if step%10==0:
            env.progress_path.write_text(json.dumps(dict(case=asdict(case),step=step,
                phase=teacher.state,sim_stamp=env.scan_stamp,label_valid=valid,
                relocalizing=teacher.relocalization is not None,world_pose_privileged=env.pose.tolist()),indent=2)+'\n')
        if handoff:
            reason='success' if goal_check(case,env.pose) else 'handoff_world_pose_mismatch';break
        if stalled:reason='no_motion_watchdog';break
        if teacher.failure:reason=teacher.failure;break
        if abs(env.roll)>.55 or abs(env.pitch)>.65:reason='excessive_tilt';break
        hold_steps=hold_steps+1 if teacher.relocalization is not None else 0
        blocked_steps=blocked_steps+1 if teacher.obstacle is not None or blocked else 0
        if hold_steps>50:reason='geometry_hold_timeout';break
        if blocked_steps>30:reason='persistent_obstacle';break
        env._publish(executed);env._drain(.02);env._advance(.2);previous=executed.copy();last_probe=command.copy()
    env._publish((0.,0.));env._drain();env._advance(.4)
    velocity=env.odom.twist.twist
    stationary=abs(velocity.linear.x)<.02 and abs(velocity.angular.z)<.03
    if reason=='success' and not stationary:reason='final_stop_unconfirmed'
    return frames,dict(case=asdict(case),success=reason=='success',reason=reason,frames=len(frames),
        valid_labels=sum(row['label_valid'] for row in frames),events=events,
        disturbance_applied=disturbed,final_world_pose=env.pose.tolist(),
        final_stationary=stationary,final_measured_velocity=[velocity.linear.x,velocity.angular.z],
        scan_geometry=dict(angle_min=env.scan.angle_min,angle_increment=env.scan.angle_increment,
            range_min=env.scan.range_min,range_max=env.scan.range_max,rays=len(env.scan.ranges)),
        actuator_inhibit_applied=inhibit_until is not None, contacts_instrumented=False,reset_and_rollout='Physical Gazebo rough world, synchronized 0.2 simulated-second steps')


def cases():
    result=[]
    for group,(family,shift,axis) in enumerate((('incoming_correction',.18,'incoming_retreat'),
            ('lateral_correction',.06,'lateral'),('temporary_actuator_inhibit',0.,'lateral'))):
        for i,task in enumerate(TASKS):
            result.append(Case(task,20300000+group*100+i,'train',family,(-6.18,0.,6.18)[(group+i)%3],
                0.,0.,.2,shift,axis))
    for i,task in enumerate(('RIGHT_ENTER','LEFT_ENTER')):
        result.append(Case(task,20301000+i,'stress','sustained_actuator_inhibit',0.,0.,0.,.2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);planned=cases()
    if args.smoke:planned=planned[:1]
    spec=contract();spec['hashes']['rl_junction/collect_correction_demos.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta=dict(contract=spec,plan=[asdict(c) for c in planned],episodes=[],
        note='New training corrections; original validation/test remain excluded from training. Actuator inhibition is synthetic software suppression, not friction/slip randomization.')
    for name in spec['hashes']:
        if name.endswith('.stl'):continue
        target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    path=args.output/'dataset.json'
    def persist():path.write_text(json.dumps(meta,indent=2)+'\n')
    persist();stop=threading.Event();env=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        env=transport();env.progress_path=args.output/'progress.json'
        for case in planned:
            if stop.is_set():break
            print('START',case.task,case.family,flush=True)
            try:frames,result=correction_episode(env,case,300,stop)
            except Exception as exc:
                frames=getattr(env,'current_case_frames',[])
                result=dict(case=asdict(case),success=False,reason='infrastructure_fault',error=repr(exc),
                    frames=len(frames),valid_labels=sum(r['label_valid'] for r in frames),bc_eligible=False)
                save_episode(args.output,case,frames,result);meta['episodes'].append(result)
                meta['infrastructure_fault']=repr(exc);persist();raise
            result['bc_eligible']=bool(result['success'] and case.split=='train' and result['valid_labels']>=10
                and (not case.turn_disturbance_m or result['disturbance_applied'])
                and ('actuator_inhibit' not in case.family or result['actuator_inhibit_applied']))
            save_episode(args.output,case,frames,result);meta['episodes'].append(result);persist()
            print('END',case.task,case.family,result['reason'],flush=True)
    finally:
        if env is not None:env.finish()
        meta['complete']=len(meta['episodes'])==len(planned) and not stop.is_set() and not meta.get('infrastructure_fault');persist()


if __name__=='__main__':main()
