"""Full local-junction actor with unchanged route FSM and external safety gates."""
from collections import deque
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rl_junction'))
from full_junction_bc import HISTORY, LOCAL_PHASES
from rough_demo_common import observation
from navigation import wrap_angle
from lidar_patrol_right_then_left import RightThenLeftPatrolController, incoming_corridor


class FullJunctionController(RightThenLeftPatrolController):
    def __init__(self,cfg,policy):
        super().__init__(cfg);self.policy=policy;self.history=deque(maxlen=HISTORY)
        self.previous=np.zeros(2);self.measured=np.zeros(2);self.measured_stamp=None
        self.observation_stamp=None;self.observation_task=None;self.full_calls=0
        self.full_owner='FSM';self.full_teacher_command=np.zeros(2);self.full_last_input=None;self.full_last_raw=np.zeros(2)
        self.no_motion_frames=0;self.last_requested=np.zeros(2)

    def observe_motion(self,v,w,stamp):
        values=np.asarray([v,w],dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(stamp):raise ValueError('Invalid odometry')
        self.measured=values;self.measured_stamp=float(stamp)

    def commit(self,executed):
        values=np.asarray(executed,dtype=float)
        if values.shape!=(2,) or not np.isfinite(values).all():raise ValueError('Invalid executed action')
        self.previous=values.copy()

    def project_local(self,action):
        """Retain geometric turn-admission/confirmation constraints outside BC.

        Only remove forbidden components; never substitute a teacher action.
        All projections are logged as interventions, not unassisted policy wins.
        """
        action=np.asarray(action,dtype=float).copy()
        if self.tracker is None:return action
        incoming=float(self.tracker.incoming_axis);centre=self.tracker.geometry.centre
        if self.state in ('CENTER','RETURN_CENTER'):
            direction=np.array([math.cos(incoming),math.sin(incoming)])
            normal=np.array([-direction[1],direction[0]])
            along,lateral=float(centre@direction),float(centre@normal)
            in_region=np.linalg.norm(centre)<=self.cfg.junction_centre_tolerance
            sign=-1 if along<0 else 1
            heading=wrap_angle(incoming if in_region else incoming+sign*np.clip(math.atan2(lateral,max(abs(along),.08)),-.55,.55))
            approach=self.main_direction if self.state=='CENTER' else -1
            speed=.05 if self.turn_reposition_attempts or sign!=approach else .08 if approach*along<.3 else .12
            action[0]=np.clip(action[0],-speed,speed)
            if in_region or abs(heading)>.3:action[0]=0.
            action[1]=np.clip(action[1],-.3,.3)
            if action[1]*heading<0 or abs(heading)<=self.cfg.yaw_tolerance:action[1]=0.
        elif self.state in ('TURN_LEFT','TURN_MAIN'):
            branch=-1 if self.pass_side=='RIGHT' else 1
            turn=branch if self.state=='TURN_LEFT' else -branch
            heading=wrap_angle(incoming+turn*math.pi/2)
            if abs(heading)<=self.cfg.yaw_tolerance:action[:]=0.
            elif action[1]*heading<0:action[1]=0.
        return action

    def command(self,features,stamp,*,allow_retreat=True):
        before=self.state;self.full_owner='FSM';self.full_last_input=None
        teacher=np.asarray(super().command(features,stamp,allow_retreat=allow_retreat))
        self.full_teacher_command=teacher.copy()
        returning=before in ('REVERSE_LEFT','RETURN_CENTER','TURN_MAIN','EXIT_MAIN','RETURN_RIGHT')
        task=self.pass_side+('_RETURN' if returning else '_ENTER')
        if task!=self.observation_task:
            self.history.clear();self.observation_task=task
        tracker=self.tracker if self.tracker is not None and abs(self.tracker.stamp-stamp)<1e-5 else None
        corridor=incoming_corridor(features,self.cfg)
        exit_corridor=(corridor if (self.state in self.ENTRY_STATES and self.entry_phase=='WALL_CONFIRM')
                       or (before in self.ENTRY_STATES and self.state in self.CORRIDOR_STATES) else None)
        dt=stamp-self.observation_stamp if self.observation_stamp is not None else .2
        self.observation_stamp=stamp
        if dt<=0:return np.zeros(2)
        fresh=self.measured_stamp is not None and abs(self.measured_stamp-stamp)<=.15
        x=observation(features,self.cfg,task,tracker,exit_corridor,self.measured,self.previous,dt)
        if not self.history:
            for _ in range(HISTORY-1):self.history.append(x.copy())
        self.history.append(x)
        local=before in LOCAL_PHASES and before==self.state
        if not local or self.failure or self.external_paused or self.relocalization is not None or self.obstacle is not None:
            self.last_requested=teacher.copy();self.no_motion_frames=0
            self.full_owner='FSM_SAFETY' if local else 'FSM';return teacher
        if not fresh or x[153]<.5:
            self.full_owner='SAFETY_HOLD';self.last_requested=np.zeros(2)
            self.diagnostics.update(full_owner=self.full_owner,full_reason='ODOM_STALE' if not fresh else 'GEOMETRY_INVALID')
            return np.zeros(2)
        demanded=abs(self.last_requested[0])>.025 or abs(self.last_requested[1])>.1
        stationary=abs(self.measured[0])+.2*abs(self.measured[1])<.008
        self.no_motion_frames=self.no_motion_frames+1 if demanded and stationary else 0
        if self.no_motion_frames>=8:
            self.full_owner='SAFETY_HOLD';return self.fail('full_no_motion_watchdog')
        try:
            self.full_last_input=np.asarray(self.history,dtype=np.float32)
            raw=np.asarray(self.policy(self.full_last_input),dtype=float)
            if raw.shape!=(2,) or not np.isfinite(raw).all():raise ValueError('Invalid full policy action')
            angular=min(self.cfg.turn_speed,.25) if self.turn_reposition_attempts else self.cfg.turn_speed
            request=np.clip(raw,[-min(.4,self.cfg.cruise_speed),-angular],[min(.4,self.cfg.cruise_speed),angular])
        except (ValueError,RuntimeError) as exc:
            self.diagnostics['full_error']=str(exc);self.full_owner='SAFETY_HOLD'
            return self.fail('full_policy_fault')
        unprojected=request.copy();request=self.project_local(request)
        self.full_calls+=1;self.full_owner='FULL_BC';self.full_last_raw=raw.copy();self.last_requested=request.copy()
        self.diagnostics.update(full_projection_changed=bool(not np.allclose(request,unprojected,atol=1e-8)),
            full_before_projection=unprojected.tolist())
        self.diagnostics.update(full_owner=self.full_owner,full_requested=request.tolist())
        return self._safe(request,features)
