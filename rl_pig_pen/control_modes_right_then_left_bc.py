#!/usr/bin/env python3
"""Separate BC turn-only entry; original straight-line FSM remains in charge."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rl_junction'))
from turn_bc import TurnPolicy
from control_modes_right_then_left import ModeControlNode
from fsm_right_then_left import main
from lidar_patrol_right_then_left import RightThenLeftPatrolController


class BCTurnNode(ModeControlNode):
    def create_controller(self,cfg):
        if not hasattr(self,'bc_policy'):
            model=self.declare_parameter('bc_model','').value
            if not model:raise ValueError('bc_model must point to a validated turn-only checkpoint')
            self.bc_policy=TurnPolicy(model)
            self.get_logger().info('混合 BC：直線／置中／交接由 FSM 控制；TURN 角速度由模型控制')
        return RightThenLeftPatrolController(cfg,turn_policy=self.bc_policy)


if __name__=='__main__':main(node_factory=BCTurnNode)
