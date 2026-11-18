#!/usr/bin/env python3
"""Collect diversified full-junction FSM demonstrations on physical rough terrain.

Only domain 47 / partition 4wd_env_47 is permitted. Existing models untouched.
World pose is privileged validation data, never part of actor observations.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from rough_demo_common import ROOT, TASKS, Case, plan, observation, goal_check, contract
sys.path.insert(0,str(ROOT/'scripts'))
from patrol_retest_monitor import pose_record
from navigation import load_config, safe_command, wrap_angle
from lidar_patrol_right_then_left import RightThenLeftPatrolController, incoming_corridor


class Teacher(RightThenLeftPatrolController):
    def _safe(self, action, features):
        result=super()._safe(action,features)
        self.last_safety=(np.asarray(action).copy(),np.asarray(result).copy())
        return result


def transport():
    import rclpy
    from gazebo_env import JunctionEnv
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from rosgraph_msgs.msg import Clock
    from ros_gz_interfaces.srv import ControlWorld, SetEntityPose

    class Transport(JunctionEnv):
        """Reuse only proven ROS transport/stepping, never old task/reset/scoring."""
        def __init__(self):
            if os.environ.get('ROS_DOMAIN_ID')!='47' or os.environ.get('IGN_PARTITION')!='4wd_env_47':
                raise RuntimeError('Requires Gazebo domain47 / partition4wd_env_47')
            self.context=rclpy.context.Context();rclpy.init(context=self.context)
            self.node=rclpy.create_node('rough_demonstration_collector',context=self.context)
            self.executor=SingleThreadedExecutor(context=self.context);self.executor.add_node(self.node)
            self.pub=self.node.create_publisher(Twist,'/cmd_vel',1)
            self.control_client=self.node.create_client(ControlWorld,'/world/pig_pen_16units_world/control')
            self.pose_client=self.node.create_client(SetEntityPose,'/world/pig_pen_16units_world/set_pose')
            self.scan=self.odom=self.pose=None;self.scan_stamp=self.odom_stamp=self.pose_stamp=-1.;self.sim_time=0.
            self.node.create_subscription(Clock,'/clock',self._clock,qos_profile_sensor_data)
            self.node.create_subscription(LaserScan,'/scan',self._scan,qos_profile_sensor_data)
            self.node.create_subscription(Odometry,'/odom',self._odom,qos_profile_sensor_data)
            self.reader_stop=threading.Event();self.roll=self.pitch=0.;self.pose_z=0.
            self.reader_error=None
            self.process=subprocess.Popen(['ign','topic','-e','-t','/world/pig_pen_16units_world/dynamic_pose/info'],
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
            self.thread=threading.Thread(target=self.read_pose,daemon=True);self.thread.start()

        def reset(self,*args,**kwargs):
            raise RuntimeError('Legacy flat-world reset disabled; use reset_case')

        def step(self,*args,**kwargs):
            raise RuntimeError('Legacy three-task scoring disabled')

        def read_pose(self):
            kind=None;block=[];depth=0;stamp=-1.
            try:
                for line in self.process.stdout:
                    if self.reader_stop.is_set():break
                    if depth==0 and line.rstrip() in ('header {','pose {'):
                        kind=line.split()[0];block=[line];depth=1;continue
                    if depth:
                        block.append(line);depth+=line.count('{')-line.count('}')
                        if depth==0:
                            text=''.join(block)
                            if kind=='header':
                                values={k:int(v) for k,v in re.findall(r'\b(sec|nsec): (\d+)',text)}
                                stamp=values.get('sec',0)+values.get('nsec',0)*1e-9
                            else:
                                value=pose_record(text)
                                if value and stamp>=0:
                                    self.pose=np.array([value['x'],value['y'],value['yaw']])
                                    self.pose_z=value['z'];self.roll=value['roll'];self.pitch=value['pitch']
                                    self.pose_stamp=stamp
            except Exception as exc:self.reader_error=str(exc)

        def teleport(self,x,y,yaw,z):
            request=SetEntityPose.Request();request.entity.name='wheeltec_mini';request.entity.type=request.entity.MODEL
            request.pose.position.x=float(x);request.pose.position.y=float(y);request.pose.position.z=float(z)
            request.pose.orientation.z=math.sin(yaw/2);request.pose.orientation.w=math.cos(yaw/2)
            self._service(self.pose_client,request)

        def reset_case(self,case):
            self.cfg=replace(load_config(ROOT/'rl_pig_pen/patrol_config_right_then_left.json'),
                cruise_speed=case.cruise_m_s,control_dt=.5,vision_required=False)
            self._assert_exclusive();self._publish((0.,0.));self._drain()
            self._control();self._drain(.2)
            self.teleport(*case.start());self._drain()
            for _ in range(5):self._advance(.2)
            start=case.start()
            if np.linalg.norm(self.pose[:2]-start[:2])>.05 or abs(wrap_angle(self.pose[2]-start[2]))>.06:
                raise RuntimeError('Teleport did not settle at requested pose')
            velocity=self.odom.twist.twist
            if abs(velocity.linear.x)>.02 or abs(velocity.angular.z)>.03:
                raise RuntimeError('Reset robot is not stationary')

        def finish(self):
            self._publish((0.,0.));self._drain()
            try:self._advance(.2)
            except RuntimeError:pass
            self.reader_stop.set();self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.thread.join(timeout=2);self.executor.shutdown();self.node.destroy_node();self.context.shutdown()
    return Transport()


def episode(env,case,limit,stop):
    env.current_case_frames=[]
    env.reset_case(case)
    teacher=Teacher(env.cfg);teacher.pass_side=case.task.split('_')[0]
    teacher.main_direction=1 if teacher.pass_side=='RIGHT' else -1
    teacher.state='MAIN' if case.task.endswith('ENTER') else 'REVERSE_LEFT'
    target='LEFT_OUTBOUND' if case.task.endswith('ENTER') else 'MAIN'
    previous=np.zeros(2);frames=[];events=[];disturbed=False;hold_steps=0;blocked_steps=0;reason='step_limit'
    env.current_case_frames=frames
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
        handoff=teacher.state==target and step>0
        valid=bool(not teacher.failure and teacher.relocalization is None and teacher.obstacle is None
                   and not teacher.external_paused and not blocked and not handoff
                   and x[153]>.5 and before==teacher.state)
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
        if teacher.failure:reason=teacher.failure;break
        if abs(env.roll)>.55 or abs(env.pitch)>.65:reason='excessive_tilt';break
        hold_steps=hold_steps+1 if teacher.relocalization is not None else 0
        blocked_steps=blocked_steps+1 if teacher.obstacle is not None or blocked else 0
        if hold_steps>50:reason='geometry_hold_timeout';break
        if blocked_steps>30:reason='persistent_obstacle';break
        env._publish(executed);env._drain(.02);env._advance(.2);previous=executed.copy()
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
        contacts_instrumented=False,reset_and_rollout='Physical Gazebo rough world, synchronized 0.2 simulated-second steps')


def save_episode(output,case,frames,result):
    name=f'{case.task.lower()}_{case.seed}'
    if frames:
        mapping={'observations':'observation','teacher_actions':'teacher_action','executed_actions':'executed_action',
            'safety_requests_m_s_rad_s':'safety_request','safety_request_available':'request_available',
            'label_valid':'label_valid','sim_stamps':'stamp','sim_control_stamps':'clock','phases':'phase','next_phases':'next_phase',
            'world_poses_privileged':'world_pose','world_z_privileged':'z','roll_privileged':'roll',
            'pitch_privileged':'pitch','raw_ranges':'raw','relocalizing':'relocalizing','obstacle':'obstacle','turn_retries':'retry'}
        arrays={key:np.asarray([row[value] for row in frames]) for key,value in mapping.items()}
        file=output/(name+'.npz');np.savez_compressed(file,**arrays)
        result.update(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases-per-task',type=int,default=8);p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261116);p.add_argument('--smoke',action='store_true')
    p.add_argument('--case-index',type=int,help='Run one planned episode for targeted validation')
    args=p.parse_args();cases=plan(args.seed,args.cases_per_task)
    if args.max_steps<1 or args.seed<0:p.error('max-steps must be positive and seed nonnegative')
    if args.smoke:cases=cases[:1]
    if args.case_index is not None:
        if args.smoke or not 0<=args.case_index<len(cases):p.error('Invalid case-index or conflicting smoke option')
        cases=[cases[args.case_index]]
    args.output.mkdir(parents=True,exist_ok=False)
    metadata=dict(contract=contract(),plan=[asdict(case) for case in cases],episodes=[],
        note='New full-junction dataset, not directly compatible with the turn-only pilot. Failed attempts preserved and excluded from BC eligibility.')
    baseline=ROOT/'logs/turn_bc/20261116_pilot/model_v2/model.pt'
    if baseline.exists():metadata['preserved_baseline_model']=dict(path=str(baseline),sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    for name in metadata['contract']['hashes']:
        if name.endswith('.stl'):continue
        target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stopping=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.set())
    env=None
    try:
        env=transport()
        env.progress_path=args.output/'progress.json'
        for case in cases:
            if stopping.is_set():break
            print('START',json.dumps(asdict(case)),flush=True)
            started=time.monotonic()
            try:frames,result=episode(env,case,args.max_steps,stopping)
            except Exception as exc:
                frames=getattr(env,'current_case_frames',[])
                result=dict(case=asdict(case),success=False,bc_eligible=False,reason='infrastructure_fault',
                    error=str(exc),frames=len(frames),valid_labels=sum(row['label_valid'] for row in frames))
                metadata['infrastructure_fault']=str(exc)
                save_episode(args.output,case,frames,result);metadata['episodes'].append(result)
                raise
            result['wall_seconds']=time.monotonic()-started
            result['bc_eligible']=bool(result['success'] and result['valid_labels']>=10 and
                (not case.turn_disturbance_m or result['disturbance_applied']))
            save_episode(args.output,case,frames,result);metadata['episodes'].append(result)
            (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('END',json.dumps(result),flush=True)
    finally:
        if env is not None:env.finish()
        metadata['complete']=len(metadata['episodes'])==len(cases) and not stopping.is_set() and not metadata.get('infrastructure_fault')
        metadata['accepted_episodes']=sum(e.get('bc_eligible',False) for e in metadata['episodes'])
        (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
