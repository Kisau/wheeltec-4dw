#!/usr/bin/env python3
"""Independent Gazebo world-pose audit; never publishes motion commands."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from patrol_retest_monitor import pose_record


def main():
    import rclpy
    from std_msgs.msg import String
    if os.environ.get('ROS_DOMAIN_ID')!='48' or os.environ.get('IGN_PARTITION')!='4wd_env_48':
        raise SystemExit('Requires domain 48 and partition 4wd_env_48')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    poses=(args.output/'world_poses.jsonl').open('w',buffering=1)
    latest={};checks=[];last_count=1;minimum=100.;max_tilt=0.
    # Exact solid wall rectangles in world XY, independent of controller odometry.
    import xml.etree.ElementTree as ET
    world=ET.parse(Path(__file__).resolve().parents[1]/'worlds/u_corridor_flat.world')
    walls=[]
    for m in world.findall('.//world/model'):
        if m.get('name')=='flat_ground':continue
        x,y,*_=map(float,m.findtext('pose').split())
        sx,sy,_=map(float,m.findtext('link/collision/geometry/box/size').split())
        walls.append((x-sx/2,y-sy/2,x+sx/2,y+sy/2))
    child=subprocess.Popen(['ign','topic','-e','-t','/world/u_corridor_world/dynamic_pose/info'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    def read():
        nonlocal minimum,max_tilt
        block=[];depth=0;last_record=0
        for line in child.stdout:
            if line.rstrip()=='pose {' and depth==0:block=[line];depth=1;continue
            if depth:
                block.append(line);depth+=line.count('{')-line.count('}')
                if depth==0:
                    value=pose_record(''.join(block))
                    if value:
                        latest.update(pose=value,received=time.monotonic())
                        minimum=min(minimum,*(math.hypot(max(l-value['x'],0,value['x']-r),
                            max(d-value['y'],0,value['y']-u)) for l,d,r,u in walls))
                        max_tilt=max(max_tilt,abs(value['roll']),abs(value['pitch']))
                        if time.monotonic()-last_record>.1:
                            poses.write(json.dumps(dict(wall_time=time.time(),**value))+'\n')
                            last_record=time.monotonic()
    threading.Thread(target=read,daemon=True).start()
    route=json.loads((Path(__file__).resolve().parents[1]/'maps/u_corridor/route.json').read_text())
    rclpy.init();n=rclpy.create_node('u_corridor_world_audit');terminal=None;control=None
    def status(m):
        nonlocal last_count,terminal,control
        control=json.loads(m.data)
        if len(control['visits'])>last_count:
            name=control['visits'][-1];target=route['waypoints'][name]
            value=latest.get('pose');age=time.monotonic()-latest.get('received',0)
            error=math.hypot(value['x']-target[0],value['y']-target[1]) if value else None
            check=dict(waypoint=name,pose=value,age_s=age,error_m=error,
                       passed=bool(value and age<1 and error<.15))
            checks.append(check);print(json.dumps(check),flush=True);last_count=len(control['visits'])
        if control['state'] in ('DONE','FAILED'):terminal=time.monotonic() if terminal is None else terminal
    sub=n.create_subscription(String,'/u_corridor/status',status,10)
    started=time.monotonic()
    try:
        while time.monotonic()-started<900 and (terminal is None or time.monotonic()-terminal<1.5):
            rclpy.spin_once(n,timeout_sec=.1)
    finally:
        child.terminate();child.wait(timeout=5);poses.close()
        summary=dict(control=control,waypoint_checks=checks,minimum_centre_wall_clearance_m=minimum,
                     max_tilt_rad=max_tilt,
                     passed=bool(control and control['state']=='DONE' and len(checks)==6
                                 and all(c['passed'] for c in checks) and minimum>.4 and max_tilt<.15),
                     note='World-pose/clearance audit, not contact-sensor collision certification')
        (args.output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True);n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
