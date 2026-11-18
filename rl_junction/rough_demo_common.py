"""ROS-free four-skill data contract and diverse physical rollout plan."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'rl_pig_pen'))
from navigation import lidar_ignore_mask, wrap_angle

TASKS = ('RIGHT_ENTER', 'RIGHT_RETURN', 'LEFT_ENTER', 'LEFT_RETURN')
SCHEMA = 'rough_full_junction_demos_v1'
OBS_SIZE = 159


@dataclass(frozen=True)
class Case:
    task: str
    seed: int
    split: str
    family: str
    junction_y: float
    lateral_m: float
    yaw_offset_rad: float
    cruise_m_s: float
    turn_disturbance_m: float = 0.
    disturbance_axis: str = 'lateral'

    def start(self):
        if self.task.endswith('ENTER'):
            sign = 1 if self.task.startswith('RIGHT') else -1
            return (-self.lateral_m, self.junction_y-sign*1.2,
                    math.pi/2+self.yaw_offset_rad, .11)
        sign = 1 if self.task.startswith('RIGHT') else -1
        return (sign*1.2, self.junction_y+sign*self.lateral_m,
                (0. if sign == 1 else math.pi)+self.yaw_offset_rad, .05)


def plan(seed=20261116, cases_per_task=8):
    if not 1 <= cases_per_task <= 8:
        raise ValueError('cases_per_task must be 1..8')
    patterns = [
        ('nominal', 0., 0., .2, 0.),
        ('lateral_positive', .04, -5., .3, 0.),
        ('lateral_negative', -.04, 5., .4, 0.),
        ('yaw_positive', .02, 12., .3, 0.),
        ('yaw_negative', -.02, -12., .4, 0.),
        ('turn_recovery', 0., 0., .2, .16),
        ('heldout_mixed', .08, -8., .3, 0.),
        ('heldout_recovery', -.02, 6., .3, .18)]
    result = []
    # Interleave tasks so a partial batch still exposes all four skills.
    for index, (family, lateral, yaw, cruise, disturbance) in enumerate(patterns[:cases_per_task]):
        for task_index, task in enumerate(TASKS):
            episode_seed = seed+index*100+task_index
            rng = np.random.default_rng(episode_seed)
            result.append(Case(task, episode_seed,
                'train' if index < 6 else 'validation' if index == 6 else 'test',
                family, (-6.18, 0., 6.18)[(index+task_index) % 3],
                lateral+float(rng.uniform(-.004,.004)),
                math.radians(yaw+float(rng.uniform(-.4,.4))), cruise, disturbance,
                'incoming_retreat' if disturbance else 'lateral'))
    return result


def observation(features, cfg, task, tracker, corridor, measured, previous, dt):
    ignored = lidar_ignore_mask(features.angles, cfg)
    bins = np.minimum((np.mod(features.angles+math.pi,2*math.pi)*72/(2*math.pi)).astype(int),71)
    ranges, usable = np.ones(72), np.zeros(72)
    for i in range(72):
        take = (bins == i) & ~ignored
        if np.any(take):
            ranges[i] = np.clip(np.min(features.ranges[take])/10.,0.,1.)
            usable[i] = 1.
    centre, centre_valid = np.zeros(2), 0.
    heading, heading_valid = 0., 0.
    if tracker is not None:
        centre = np.clip(tracker.geometry.centre/1.5,-2.,2.)
        direction = {'RIGHT_ENTER':-1,'RIGHT_RETURN':1,'LEFT_ENTER':1,'LEFT_RETURN':-1}[task]
        heading = wrap_angle(tracker.incoming_axis+direction*math.pi/2)
        centre_valid = heading_valid = 1.
    elif corridor is not None:
        heading, heading_valid = float(corridor[0]), 1.
    x = np.r_[ranges,usable,np.eye(4)[TASKS.index(task)],centre,
              math.sin(heading),math.cos(heading),centre_valid,heading_valid,
              np.clip(np.asarray(measured)/[.4,.6],-2.,2.),
              np.clip(np.asarray(previous)/[.4,.6],-1.,1.),dt/.2]
    if x.shape != (OBS_SIZE,) or not np.isfinite(x).all():
        raise ValueError('Invalid causal junction observation')
    return x.astype(np.float32)


def goal_check(case, pose):
    x,y,yaw = pose
    if case.task.endswith('ENTER'):
        direction = 1 if case.task.startswith('RIGHT') else -1
        target_yaw = 0. if direction == 1 else math.pi
        return bool(direction*x > .72 and abs(y-case.junction_y)<.15 and
                    abs(wrap_angle(yaw-target_yaw))<math.radians(10))
    direction = 1 if case.task.startswith('RIGHT') else -1
    return bool(direction*(y-case.junction_y)>.72 and abs(x)<.15 and
                abs(wrap_angle(yaw-math.pi/2))<math.radians(10))


def contract():
    names = ['rl_junction/rough_demo_common.py','rl_junction/collect_rough_demos.py',
        'rl_junction/gazebo_env.py','scripts/patrol_retest_monitor.py',
        'rl_pig_pen/lidar_patrol_right_then_left.py','rl_pig_pen/navigation.py',
        'rl_pig_pen/lidar_geometry.py','rl_pig_pen/junction_localization.py',
        'rl_pig_pen/patrol_config_right_then_left.json','worlds/pig_pen_16units_rough.world',
        'worlds/pig_pen_16units_rough.terrain.json',
        'turn_on_wheeltec_robot/urdf/four_wheel_diff_bs_robot.urdf']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'worlds/meshes/rough_pen').glob('*_gravel.stl'))]
    return dict(schema=SCHEMA,observation_size=OBS_SIZE,dt_sim_s=.2,
        action_mapping='v=.4*a0; w=.6*a1; bounds [-1,1]',tasks=TASKS,
        world_pose_usage='reset/perturbation/scoring only; never policy observation',
        phase_usage='metadata/label mask only; not policy observation',
        terrain='fixed physical embedded gravel, raised floor, ramps and posts; no per-episode friction/mass randomization',
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
