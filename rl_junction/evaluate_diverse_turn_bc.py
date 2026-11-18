#!/usr/bin/env python3
"""Simulation-only paired evaluation of the original and diversified turn actors.

FSM controls all straight, centring, recovery, transitions and safety checks.
World pose is used only for reset, synthetic disturbance and independent scoring.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time
import numpy as np
from collect_rough_demos import Teacher, transport
from rough_demo_common import ROOT, plan, goal_check, contract
from navigation import safe_command
from turn_bc import TurnPolicy


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class MeasuredPolicy:
    def __init__(self,policy):self.policy=policy;self.calls=[]
    def __call__(self,features,controller,heading):
        start=time.perf_counter();w=self.policy(features,controller,heading)
        self.calls.append(dict(stamp=controller.last_stamp,phase=controller.state,
            linear_x=0.,raw_angular_z=w,heading=heading,centre=controller.tracker.geometry.centre.tolist(),
            retry=controller.turn_reposition_attempts,inference_ms=(time.perf_counter()-start)*1000))
        return w


def evaluate_case(env,case,policy,folder,limit,stop):
    folder.mkdir();measured=MeasuredPolicy(policy);env.reset_case(case)
    teacher=Teacher(env.cfg,turn_policy=measured);teacher.pass_side=case.task.split('_')[0]
    teacher.main_direction=1 if teacher.pass_side=='RIGHT' else -1
    teacher.state='MAIN' if case.task.endswith('ENTER') else 'REVERSE_LEFT'
    target='LEFT_OUTBOUND' if case.task.endswith('ENTER') else 'MAIN'
    disturbed=False;events=[];rows=[];raw=[];blocked_steps=hold_steps=0;reason='step_limit'
    with gzip.open(folder/'decisions.jsonl.gz','wt') as log:
        for step in range(limit):
            if stop.is_set():reason='interrupted';break
            if case.turn_disturbance_m and not disturbed and teacher.state in ('TURN_LEFT','TURN_MAIN'):
                x,y,yaw=env.pose.copy();approach=teacher.main_direction if teacher.state=='TURN_LEFT' else -1
                dx=-approach*abs(case.turn_disturbance_m)
                env._publish((0.,0.));env._drain()
                env.teleport(x+math.cos(yaw)*dx,y+math.sin(yaw)*dx,yaw,env.pose_z)
                env._drain();env._advance(.2);disturbed=True
                events.append(dict(step=step,stamp=env.scan_stamp,body_offset_m=[dx,0.],
                    type='synthetic_turn_pose_shift',note='Not a physical slip model'))
            features=env._features();before=teacher.state;call_start=len(measured.calls)
            command=np.asarray(teacher.command(features,env.scan_stamp),dtype=float)
            executed,blocked=safe_command(command,features,env.cfg,allow_reverse=True)
            owned=len(measured.calls)>call_start
            row=dict(step=step,stamp=env.scan_stamp,phase=before,next_phase=teacher.state,
                command=command.tolist(),executed=executed.tolist(),model_called=owned,
                relocalizing=teacher.relocalization is not None,obstacle=teacher.obstacle is not None,
                guard_blocked=bool(blocked),retry=teacher.turn_reposition_attempts,
                world_pose_privileged=env.pose.tolist(),z_privileged=env.pose_z,
                roll_privileged=env.roll,pitch_privileged=env.pitch,diagnostics=teacher.diagnostics.copy())
            if owned and (before not in ('TURN_LEFT','TURN_MAIN') or abs(command[0])>1e-8):
                raise RuntimeError('BC took ownership outside pure turn')
            if (before==teacher.state and before in ('TURN_LEFT','TURN_MAIN') and
                    abs(command[1])>1e-7 and not owned):
                raise RuntimeError('Nonzero turn bypassed model')
            rows.append(row);raw.append(np.asarray(env.scan.ranges,dtype=np.float32).copy())
            log.write(json.dumps(row,default=plain)+'\n');log.flush()
            if step%10==0:
                env.progress_path.write_text(json.dumps(dict(case=asdict(case),step=step,phase=teacher.state,
                    model=str(policy.metadata['dataset']),world_pose_privileged=env.pose.tolist()),indent=2)+'\n')
            if teacher.state==target and step>0:
                reason='success' if goal_check(case,env.pose) else 'handoff_world_pose_mismatch';break
            if teacher.failure:reason=teacher.failure;break
            if abs(env.roll)>.55 or abs(env.pitch)>.65:reason='excessive_tilt';break
            hold_steps=hold_steps+1 if teacher.relocalization is not None else 0
            blocked_steps=blocked_steps+1 if teacher.obstacle is not None or blocked else 0
            if hold_steps>50:reason='geometry_hold_timeout';break
            if blocked_steps>30:reason='persistent_obstacle';break
            env._publish(executed);env._drain(.02);env._advance(.2)
    env._publish((0.,0.));env._drain();env._advance(.4)
    velocity=env.odom.twist.twist
    stationary=abs(velocity.linear.x)<.02 and abs(velocity.angular.z)<.03
    if reason=='success' and not stationary:reason='final_stop_unconfirmed'
    if reason=='success' and not measured.calls:reason='no_model_calls'
    (folder/'model_calls.json').write_text(json.dumps(measured.calls,indent=2)+'\n')
    np.savez_compressed(folder/'scans.npz',ranges=np.asarray(raw),stamps=[r['stamp'] for r in rows])
    result=dict(case=asdict(case),success=reason=='success',reason=reason,steps=len(rows),
        bc_calls=len(measured.calls),ownership_audit_passed=True,disturbance_applied=disturbed,events=events,
        max_retries=max((r['retry'] for r in rows),default=0),
        relocalization_frames=sum(r['relocalizing'] for r in rows),
        safety_intervention_frames=sum(r['obstacle'] or r['guard_blocked'] for r in rows),
        final_world_pose=env.pose.tolist(),final_stationary=stationary,
        final_measured_velocity=[velocity.linear.x,velocity.angular.z],
        inference_p95_ms=float(np.percentile([c['inference_ms'] for c in measured.calls],95)) if measured.calls else None,
        scan_geometry=dict(angle_min=env.scan.angle_min,angle_increment=env.scan.angle_increment,
            range_min=env.scan.range_min,range_max=env.scan.range_max),
        contacts_instrumented=False)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=300);args=p.parse_args()
    if args.max_steps<1:p.error('max-steps must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    models={'new':TurnPolicy(args.model),'baseline':TurnPolicy(args.baseline)}
    hashes={name:dict(path=str(path.resolve()),sha256=digest(path))
            for name,path in [('new',args.model),('baseline',args.baseline)]}
    cases=plan(seed=20261117)
    trials=[('new',c) for c in cases[:4]]+[(name,c) for c in cases[-4:] for name in ('baseline','new')]
    meta=dict(contract=contract(),evaluator_sha256=digest(__file__),models=hashes,
        plan=[dict(model=name,case=asdict(c)) for name,c in trials],episodes=[],
        note='Four new-model nominal cases plus four paired baseline/new recovery cases, 20261117 seeds. Fixed terrain and small evaluation; not a learned-policy generalization certificate.')
    path=args.output/'evaluation.json'
    def persist():path.write_text(json.dumps(meta,indent=2)+'\n')
    persist();stop=threading.Event();env=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        env=transport();env.progress_path=args.output/'progress.json'
        for index,(name,case) in enumerate(trials):
            if stop.is_set():break
            folder=args.output/f'{index:02d}_{name}_{case.task.lower()}'
            print('START',index,name,case.task,case.family,flush=True);started=time.monotonic()
            try:result=evaluate_case(env,case,models[name],folder,args.max_steps,stop)
            except Exception as exc:
                meta['infrastructure_fault']=repr(exc);persist();raise
            result.update(model=name,folder=folder.name,wall_seconds=time.monotonic()-started)
            meta['episodes'].append(result);persist();print('END',index,name,result['reason'],flush=True)
    finally:
        if env is not None:env.finish()
        meta['complete']=len(meta['episodes'])==len(trials) and not stop.is_set() and not meta.get('infrastructure_fault')
        meta['models_unchanged']=all(digest(info['path'])==info['sha256'] for info in hashes.values())
        persist()


if __name__=='__main__':main()
