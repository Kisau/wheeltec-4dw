#!/usr/bin/env python3
"""Separate simulation-only full-junction BC entry; original entries untouched."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rl_junction'))
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from full_junction_bc import FullPolicy
from full_junction_controller import FullJunctionController
from control_modes_right_then_left import ModeControlNode
from fsm_right_then_left import main


class FullBCNode(ModeControlNode):
    def __init__(self):
        if os.environ.get('ROS_DOMAIN_ID')!='47' or os.environ.get('IGN_PARTITION')!='4wd_env_47':raise RuntimeError('Experimental full BC is simulation-only on domain 47')
        super().__init__()
        self.odom_sub=self.create_subscription(Odometry,'/odom',self.on_full_odom,qos_profile_sensor_data)
    def create_controller(self,cfg):
        if not hasattr(self,'full_policy'):
            path=self.declare_parameter('full_bc_model','').value
            if not path:raise ValueError('full_bc_model must point to a validated checkpoint')
            self.full_policy=FullPolicy(path)
        return FullJunctionController(cfg,self.full_policy)
    def on_full_odom(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.controller.observe_motion(msg.twist.twist.linear.x,msg.twist.twist.angular.z,stamp)
    def publish(self,command):
        super().publish(command)
        if isinstance(self.controller,FullJunctionController):self.controller.commit(self.commanded)


if __name__=='__main__':main(node_factory=FullBCNode)
