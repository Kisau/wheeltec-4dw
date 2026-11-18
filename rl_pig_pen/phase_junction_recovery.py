"""Stopped, measured-wall recovery for the phase BC experiment only.

The original controller and trained actor contract remain unchanged. Never
integrates a command or odometry displacement to invent junction position.
"""
from copy import deepcopy
import numpy as np
from phase_junction_controller import PhaseJunctionController


class RecoveringPhaseJunctionController(PhaseJunctionController):
    def __init__(self, cfg, policy):
        super().__init__(cfg, policy)
        self.wall_recovery_candidate = None
        self.wall_recovery_frames = 0
        self.wall_identity_state = None
        self.wall_recovery_events = []

    def command(self, features, stamp, *, allow_retreat=True):
        events_before = len(self.wall_recovery_events)
        calls_before = self.full_calls
        command = super().command(features, stamp, allow_retreat=allow_retreat)
        if len(self.wall_recovery_events) != events_before:
            # Parent phase controller may compute an actor output after the
            # supervisor recovered. Discard it on this stationary handoff scan.
            self.full_calls = calls_before
            self.full_last_input = None
            self.full_last_raw = np.zeros(2)
            self.full_owner = 'SAFETY_HOLD'
            self.last_requested = np.zeros(2)
            self.diagnostics.update(full_owner='SAFETY_HOLD', full_reason='WALL_RECOVERY_HANDOFF')
            return np.zeros(2)
        return command

    def _geometry(self, features):
        # Only after a stationary, independently measured identity confirmation.
        if self.wall_identity_state == self.state and self.tracker is not None:
            geometry = self.tracker.observe_walls(features)
            if geometry is not None:
                self.diagnostics['geometry_source'] = 'confirmed_tracked_walls'
                return geometry
        return super()._geometry(features)

    def _relocalization_command(self, features, stamp):
        recovery = self.relocalization
        eligible = (recovery['reason'] == 'junction_geometry_conflict'
                    and self.state in ('CENTER', 'RETURN_CENTER')
                    and self.tracker is not None)
        fresh = self.measured_stamp is not None and abs(stamp-self.measured_stamp) <= .15
        stationary = abs(self.measured[0]) < .008 and abs(self.measured[1]) < .03
        if not eligible or not fresh or not stationary:
            self.wall_recovery_candidate = None
            self.wall_recovery_frames = 0
            return super()._relocalization_command(features, stamp)
        # A short stationary hold can rebase TIME, never the saved centre/axes.
        # Long data gaps are not repaired with this path.
        reference = self.wall_recovery_candidate or self.tracker
        if stamp-reference.stamp > self.cfg.junction_tracking_max_gap:
            self.wall_recovery_candidate = None
            self.wall_recovery_frames = 0
            return super()._relocalization_command(features, stamp)
        candidate = deepcopy(self.wall_recovery_candidate or self.tracker)
        geometry = candidate.observe_walls(features)
        if geometry is not None and candidate.update(geometry, stamp):
            self.wall_recovery_candidate = candidate
            self.wall_recovery_frames += 1
            self.diagnostics = dict(relocalization_status='waiting',
                relocalization_reason=recovery['reason'],
                relocalization_source='stationary_tracked_walls',
                relocalization_frames=self.wall_recovery_frames,
                candidate_centre=geometry.centre.tolist())
            if self.wall_recovery_frames >= self.cfg.confirm_frames:
                self.wall_identity_state = self.state
                self.wall_recovery_events.append(dict(stamp=stamp,state=self.state,
                    source='stationary_tracked_walls',frames=self.wall_recovery_frames))
                result = self._finish_relocalization(stamp, tracker=candidate)
                self.diagnostics['relocalization_source'] = 'stationary_tracked_walls'
                self.wall_recovery_candidate = None
                self.wall_recovery_frames = 0
                return result
            return np.zeros(2)
        self.wall_recovery_candidate = None
        self.wall_recovery_frames = 0
        return super()._relocalization_command(features, stamp)
