#!/usr/bin/env python3
"""Read-only telemetry and independent endpoint audit for Gazebo domain 47."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


def pose_record(block):
    if not re.search(r'\n  name: "wheeltec_mini"\n', '\n'+block):
        return None
    values = {}
    for field in ('position', 'orientation'):
        match = re.search(field+r' \{(.*?)\}', block, re.S)
        if not match:
            return None
        values[field] = {k: float(v) for k,v in re.findall(r'\b([xyzw]): ([^\n]+)',match[1])}
    p,q=values['position'],values['orientation']
    x,y,z,w=[q.get(k,0.) for k in ('x','y','z','w')]
    return dict(x=p.get('x',0.),y=p.get('y',0.),z=p.get('z',0.),
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
        pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def endpoint_check(index, pose, age):
    if pose is None or age > 2.:
        return dict(passed=False,reason='missing_or_stale_world_pose')
    side='RIGHT' if index<3 else 'LEFT'
    expected_y=(-6.18,0.,6.18,6.18,0.,-6.18)[index]
    return dict(passed=(pose['x']>2.5 if side=='RIGHT' else pose['x']< -2.5)
                and abs(pose['y']-expected_y)<.5,
                side=side,expected_y=expected_y,pose=pose,pose_age_s=age)


def main():
    import rclpy
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from rclpy.qos import qos_profile_sensor_data
    if os.environ.get('ROS_DOMAIN_ID')!='47' or os.environ.get('IGN_PARTITION')!='4wd_env_47':
        raise RuntimeError('Read-only monitor requires isolated Gazebo domain 47 / partition 4wd_env_47')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=3600.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    log=(args.output/'telemetry.jsonl').open('w',buffering=1)
    lock=threading.Lock();latest_pose=None;pose_wall=0.;last_pose_log=0.
    stop=threading.Event()
    process=subprocess.Popen(['ign','topic','-e','-t','/world/pig_pen_16units_world/dynamic_pose/info'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    def record(kind,data):
        with lock:log.write(json.dumps(dict(wall_time=time.time(),kind=kind,data=data),allow_nan=False)+'\n')
    def world_reader():
        nonlocal latest_pose,pose_wall,last_pose_log
        block=[];depth=0
        for line in process.stdout:
            if stop.is_set():break
            if line.rstrip()=='pose {' and depth==0:block=[line];depth=1;continue
            if depth:
                block.append(line);depth+=line.count('{')-line.count('}')
                if depth==0:
                    value=pose_record(''.join(block))
                    if value:
                        latest_pose=value;pose_wall=time.time()
                        if pose_wall-last_pose_log>=.2:
                            record('world_pose',value);last_pose_log=pose_wall
    thread=threading.Thread(target=world_reader,daemon=True);thread.start()
    rclpy.init();node=rclpy.create_node('patrol_retest_read_only_monitor')
    state={};visits=[];armed=False;last_command=None;last_command_wall=0.;last_scan_wall=None
    scan_gaps=[];last_scan_log=0.;last_debug_key=None;terminal_time=None
    def status(msg):
        nonlocal armed,terminal_time
        value=json.loads(msg.data);state['control']=value
        if value['mode']=='AUTO' and value['route_state']=='MAIN' and sum(value.get('side_endpoints',{}).values())==0:
            armed=True
        record('control_status',value)
        if armed:
            count=sum(value.get('side_endpoints',{}).values())
            while len(visits)<min(count,6):
                visits.append(endpoint_check(len(visits),latest_pose,time.time()-pose_wall))
            if value['route_state'] in ('DONE','FAILED') and terminal_time is None:terminal_time=time.monotonic()
    def debug(msg):
        nonlocal last_debug_key
        value=json.loads(msg.data);state['debug']=value
        key=(value.get('state'),value.get('reason'),value.get('endpoints_reached'))
        if key!=last_debug_key:
            record('patrol_debug',value);print(json.dumps(value,ensure_ascii=False),flush=True);last_debug_key=key
    def command(msg):
        nonlocal last_command,last_command_wall
        last_command=[msg.linear.x,msg.angular.z];last_command_wall=time.time()
        record('cmd_vel',last_command)
    def scan(msg):
        nonlocal last_scan_wall,last_scan_log
        now=time.time()
        if last_scan_wall is not None and now-last_scan_wall>.4:scan_gaps.append(now-last_scan_wall)
        last_scan_wall=now
        if now-last_scan_log>1:
            record('scan_timing',dict(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,rays=len(msg.ranges)))
            last_scan_log=now
    node.create_subscription(String,'/control/status',status,10)
    node.create_subscription(String,'/patrol/debug',debug,10)
    node.create_subscription(Twist,'/cmd_vel',command,10)
    node.create_subscription(LaserScan,'/scan',scan,qos_profile_sensor_data)
    started=time.monotonic();last_summary=0.
    def summary():
        done=state.get('control',{}).get('route_state')=='DONE'
        fresh_pose=latest_pose is not None and time.time()-pose_wall<2
        return dict(armed=armed,elapsed_wall_s=time.monotonic()-started,
            fsm_done=done,observed_endpoint_count=len(visits),endpoint_checks=visits,
            independent_endpoint_checks_passed=len(visits)==6 and all(v['passed'] for v in visits),
            latest=state,world_pose=latest_pose,
            return_to_start_region=bool(fresh_pose and abs(latest_pose['x'])<.2 and abs(latest_pose['y']+11.3)<.8),
            last_cmd_vel=last_command,zero_command_confirmed=bool(last_command==[0.,0.] and time.time()-last_command_wall<2),
            scan_receive_gaps_over_0_4_s=len(scan_gaps),max_observed_scan_gap_s=max(scan_gaps,default=0.),
            contacts_instrumented=False,limitation='No contact sensor: endpoint pose checks cannot certify zero physical contacts')
    try:
        while rclpy.ok() and time.monotonic()-started<args.timeout:
            rclpy.spin_once(node,timeout_sec=.1)
            now=time.monotonic()
            if now-last_summary>5:
                (args.output/'live_summary.json').write_text(json.dumps(summary(),indent=2)+'\n');last_summary=now
            if terminal_time and now-terminal_time>1.:break
        result=summary();result['monitor_end']='terminal' if terminal_time else 'timeout'
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        stop.set();process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        thread.join(timeout=2);node.destroy_node();rclpy.shutdown();log.close()


if __name__=='__main__':main()
