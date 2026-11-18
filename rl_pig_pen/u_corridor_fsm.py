"""Deterministic U-corridor route, in the spawn-relative odometry frame.

No ROS, learning, simulator world poses or dependencies in the controller.
The configured start is A, facing B; localisation is odometry, not SLAM.
"""
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clip(x, bound):
    return max(-bound, min(bound, x))


class UCorridorFSM:
    def __init__(self, side=23., top=22., speed=.4):
        if not all(math.isfinite(x) and x > 0 for x in (side, top, speed)) or speed > .5:
            raise ValueError('positive dimensions and speed <= .5 required')
        # Every leg has its own body heading, including the reverse legs.
        self.legs = [
            ('A', 'B', (0., 0.), (side, 0.), 0., 1),
            ('B', 'C', (side, 0.), (side, -top), -math.pi/2, 1),
            ('C', 'D', (side, -top), (0., -top), -math.pi, 1),
            ('D', 'C', (0., -top), (side, -top), -math.pi, -1),
            ('C', 'B', (side, -top), (side, 0.), -math.pi/2, -1),
            ('B', 'A', (side, 0.), (0., 0.), 0., -1),
        ]
        self.speed = speed
        self.index = 0
        self.state = 'ALIGN'
        self.reason = 'START'
        self.visits = ['A']
        self.stable = 0
        self.entered = None
        self.progress_time = None
        self.progress_pose = None
        self.last_now = None

    def fail(self, reason):
        self.state, self.reason = 'FAILED', reason
        return 0., 0.

    def change(self, state, now):
        self.state, self.reason = state, state
        self.entered = now
        self.stable = 0
        self.progress_time = now
        self.progress_pose = None

    def status(self):
        a, b, *_ = self.legs[self.index]
        return dict(state=self.state, reason=self.reason, leg=f'{a}->{b}',
                    direction='FORWARD' if self.index < 3 else 'REVERSE',
                    visits=self.visits.copy())

    def step(self, pose, measured, points, now):
        if self.state in ('DONE', 'FAILED'):
            return 0., 0.
        if not all(math.isfinite(x) for x in (*pose, *measured, now)):
            return self.fail('INVALID_ODOMETRY')
        if self.last_now is not None and now < self.last_now:
            return self.fail('CLOCK_RESET')
        self.last_now = now
        if self.entered is None:
            self.entered = self.progress_time = now
        if len(points) < 100:
            return self.fail('INSUFFICIENT_SCAN')
        a, b, start, goal, heading, direction = self.legs[self.index]
        x, y, yaw = pose
        error = wrap(heading-yaw)
        dx, dy = goal[0]-start[0], goal[1]-start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx/length, dy/length
        along = (x-start[0])*ux+(y-start[1])*uy
        cross = -(x-start[0])*uy+(y-start[1])*ux
        remaining = length-along
        if abs(cross) > .55:
            return self.fail('CROSS_TRACK_LIMIT')
        v = w = 0.
        if self.state == 'ALIGN':
            # Rotation requires the vehicle to have actually settled first.
            if abs(measured[0]) < .025 and abs(error) > .025:
                w = clip(1.6*error, .5)
            self.stable = self.stable+1 if abs(error) <= .025 and abs(measured[1]) < .05 else 0
            if self.stable >= 4:
                self.change('DRIVE', now)
            elif now-self.entered > 20:
                return self.fail('TURN_TIMEOUT')
        elif self.state == 'DRIVE':
            if remaining < -.15:
                return self.fail('WAYPOINT_OVERSHOOT')
            if remaining <= .035:
                self.change('SETTLE', now)
            else:
                travel_heading = math.atan2(dy, dx)
                desired = travel_heading - math.atan2(1.3*cross, .7)
                if direction < 0:
                    desired += math.pi
                steer = wrap(desired-yaw)
                if abs(steer) > .6:
                    return self.fail('HEADING_LIMIT')
                v = direction*min(self.speed, max(.04, .8*remaining))*max(.25, math.cos(steer))
                w = clip(1.8*steer, .45)
        elif self.state == 'SETTLE':
            self.stable = self.stable+1 if abs(measured[0]) < .015 and abs(measured[1]) < .025 else 0
            if self.stable >= 8 and now-self.entered >= 1.:
                self.visits.append(b)
                if self.index == len(self.legs)-1:
                    self.change('DONE', now)
                else:
                    self.index += 1
                    self.change('ALIGN', now)
            elif now-self.entered > 8:
                return self.fail('STOP_TIMEOUT')
        # Body envelope includes wheels, margin, reaction and braking distance.
        # Scan origin is the robot centre in the supplied Gazebo URDF.
        if abs(w) > .01 and any(math.hypot(px, py) < .46 for px, py in points):
            return self.fail('ROTATION_BLOCKED')
        if v:
            reach = .35 + abs(v)*.35 + v*v/(2*.5)
            sign = 1 if v > 0 else -1
            if any(-.25 < sign*px < reach and abs(py) < .34 for px, py in points):
                return self.fail('TRAVEL_BLOCKED')
        if v or w:
            if self.progress_pose is None:
                self.progress_pose, self.progress_time = pose, now
            px, py, pa = self.progress_pose
            if math.hypot(x-px, y-py) > .02 or abs(wrap(yaw-pa)) > .035:
                self.progress_pose, self.progress_time = pose, now
            elif now-self.progress_time > 3.:
                return self.fail('NO_PROGRESS')
        else:
            self.progress_pose, self.progress_time = None, now
        return v, w
