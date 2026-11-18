#!/usr/bin/env python3
"""Gazebo-only U route runner. Explicit isolated ROS domain 48 required."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rl_pig_pen'))
from u_corridor_fsm import UCorridorFSM, wrap
from u_corridor_localization import WallLocalizer


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '48':
        raise SystemExit('Simulation only: set ROS_DOMAIN_ID=48')
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speed', type=float, default=.4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    lock = open('/tmp/4wd_u_fsm_domain48.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    route = json.loads((Path(__file__).resolve().parents[1]/'maps/u_corridor/route.json').read_text())
    fsm = UCorridorFSM(route['side_length'], route['top_length'], args.speed)
    base=Path(__file__).resolve().parents[1]
    spawn=route['spawn']
    localizer=WallLocalizer(base/'worlds/u_corridor_flat.world',(spawn['x'],spawn['y'],spawn['yaw']))
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output/'telemetry.jsonl').open('w', buffering=1)
    rclpy.init()
    n = rclpy.create_node('u_corridor_fsm')
    pub = n.create_publisher(Twist, '/cmd_vel', 10)
    status = n.create_publisher(String, '/u_corridor/status', 10)
    data = {}
    stopped = False
    def stop_signal(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    def scan(m):
        data['scan_received'] = time.monotonic()
        data['scan_stamp'] = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        data['points'] = [(r*math.cos(m.angle_min+i*m.angle_increment),
                           r*math.sin(m.angle_min+i*m.angle_increment))
                          for i,r in enumerate(m.ranges)
                          if math.isfinite(r) and m.range_min <= r <= m.range_max]
    def odom(m):
        q=m.pose.pose.orientation; pos=m.pose.pose.position
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        data.update(pose=(pos.x,pos.y,yaw), measured=(m.twist.twist.linear.x,m.twist.twist.angular.z),
                    stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                    odom_received=time.monotonic())
    subs = [n.create_subscription(LaserScan,'/scan',scan,qos_profile_sensor_data),
            n.create_subscription(Odometry,'/odom',odom,qos_profile_sensor_data)]
    started = time.monotonic(); active = False; last_stamp = None; last_state = None
    last_scan_stamp=None
    final = 'INTERRUPTED'; terminal = None
    try:
        while not stopped:
            rclpy.spin_once(n, timeout_sec=.02)
            wall = time.monotonic()
            cmd = (0., 0.)
            if 'pose' in data and 'points' in data:
                if not active:
                    peers=[v.node_name for v in n.get_publishers_info_by_topic('/cmd_vel') if v.node_name != n.get_name()]
                    if peers: raise RuntimeError(f'Other velocity publishers: {peers}')
                    if math.hypot(*data['pose'][:2]) > .15 or abs(wrap(data['pose'][2])) > .08:
                        raise RuntimeError('Expected fresh spawn at A with zero relative odometry')
                    active = True
                if wall-data['scan_received'] > 1 or wall-data['odom_received'] > 1:
                    fsm.fail('SENSOR_RECEIVE_TIMEOUT')
                elif abs(data['stamp']-data['scan_stamp']) > .5:
                    fsm.fail('SENSOR_STAMP_MISMATCH')
                elif data['stamp'] != last_stamp:
                    localizer.predict(data['pose'])
                    if data['scan_stamp'] != last_scan_stamp:
                        if not localizer.correct(data['points']):
                            fsm.fail('WALL_LOCALIZATION_LOST')
                        last_scan_stamp=data['scan_stamp']
                    cmd=fsm.step(localizer.pose,data['measured'],data['points'],data['stamp'])
                    last_stamp=data['stamp']
                    record=dict(wall_time=time.time(),sim_time=data['stamp'],pose=localizer.pose,
                                raw_odometry=data['pose'],localization=localizer.quality,
                                measured=data['measured'],command=cmd,**fsm.status())
                    log.write(json.dumps(record)+'\n')
                    status.publish(String(data=json.dumps(record)))
                else:
                    continue
                state=(fsm.index,fsm.state,fsm.reason)
                if state != last_state:
                    print(json.dumps(dict(sim_time=data['stamp'],pose=localizer.pose,**fsm.status())),flush=True)
                    last_state=state
                if fsm.state in ('DONE','FAILED'):
                    cmd=(0.,0.)
                    if terminal is None: terminal=wall
                    if wall-terminal > 1.:
                        final=fsm.state
                        break
            elif wall-started > 15:
                raise RuntimeError('No scan/odometry within 15 seconds')
            msg=Twist(); msg.linear.x, msg.angular.z=cmd; pub.publish(msg)
    finally:
        if fsm.state not in ('DONE','FAILED'): fsm.reason='OPERATOR_INTERRUPT'
        for _ in range(10):
            pub.publish(Twist()); time.sleep(.03)
        summary=dict(result=final,**fsm.status(),last_pose=data.get('pose'),
                     last_measured=data.get('measured'),speed=args.speed,
                     localized_pose=localizer.pose,
                     note='LiDAR wall-map localisation + odometry; no world-pose feedback')
        (args.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        log.close(); n.destroy_node(); rclpy.shutdown(); lock.close()
    return 0 if final=='DONE' else 1


if __name__=='__main__':
    raise SystemExit(main())
