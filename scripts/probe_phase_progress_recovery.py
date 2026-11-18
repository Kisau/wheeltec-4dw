#!/usr/bin/env python3
"""Gazebo fault injection: zero actor at ~21 cm must stop then use counted FSM.

This tests the supervisor, not the learned policy's navigation success.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'rl_junction'))
from collect_rough_demos import transport
from rough_demo_common import Case
from phase_junction_controller import PhaseJunctionController
from navigation import safe_command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);env=None;rows=[];result={}
    try:
        env=transport();case=Case('RIGHT_ENTER',20600000,'stress','zero_policy_fault',0.,0.,0.,.2)
        env.reset_case(case);env.teleport(.03,-.21,np.pi/2+.05,.11);env._drain()
        for _ in range(5):env._advance(.2)
        c=PhaseJunctionController(env.cfg,lambda history:np.zeros(2));start_distance=None
        for step in range(60):
            velocity=env.odom.twist.twist;c.observe_motion(velocity.linear.x,velocity.angular.z,env.odom_stamp)
            command=c.command(env._features(),env.scan_stamp);executed,blocked=safe_command(command,env._features(),env.cfg,allow_reverse=True)
            metric=c.progress_metric()
            row=dict(step=step,stamp=env.scan_stamp,state=c.state,owner=c.full_owner,
                command=np.asarray(command).tolist(),executed=executed.tolist(),blocked=bool(blocked),
                measured=[velocity.linear.x,velocity.angular.z],metric=None if metric is None else metric[0],
                correction_count=c.correction_count,failure=c.failure)
            rows.append(row)
            (args.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
            if metric is not None and start_distance is None:start_distance=metric[0]
            if c.failure:break
            correction_rows=[r for r in rows if r['owner']=='FSM_CORRECTION']
            hold=any(r['owner']=='SAFETY_HOLD' and np.allclose(r['command'],0) for r in rows)
            progressed=metric is not None and start_distance is not None and metric[0]<start_distance-.02
            if hold and len(correction_rows)>=3 and progressed and abs(velocity.linear.x)>.02:
                result=dict(passed=True,reason='Stopped first, explicit FSM correction made measured geometric progress');break
            env._publish(executed);env._drain(.02);env._advance(.2);c.commit(executed)
        if not result:result=dict(passed=False,reason=c.failure or 'probe_step_limit')
        result.update(fault='Injected zero actor, not learned model output',rows=rows,events=c.correction_events,
            initial_metric=start_distance,final_world_pose_privileged=env.pose.tolist())
    finally:
        if env is not None:
            env._publish((0.,0.));env._drain();env._advance(.4)
            v=env.odom.twist.twist;result['final_stationary']=abs(v.linear.x)<.02 and abs(v.angular.z)<.03
            env.finish()
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result.get('passed') or not result.get('final_stationary'):raise SystemExit(1)


if __name__=='__main__':main()
