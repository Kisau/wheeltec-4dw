#!/usr/bin/env python3
"""Read completed Gazebo artifacts and audit turn-only BC ownership.

Run after the controller exits so compressed geometry logs have valid footers.
This script neither starts ROS nor sends robot commands.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TURN_STATES = ('TURN_LEFT', 'TURN_MAIN')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def turn_ownership(rows):
    segments, current, previous = [], None, None
    owned, missing = [], []
    for row in rows:
        state = row['state'] if row['state'] in TURN_STATES else None
        if state != previous:
            if current:
                segments.append(current)
            current = dict(state=state, bc_frames=0) if state else None
        previous = state
        is_bc = row['controller'].get('command_owner') == 'BC_TURN'
        if is_bc:
            owned.append(row)
            if current:
                current['bc_frames'] += 1
        if state and abs(row['command'][1]) > 1e-8 and not is_bc:
            missing.append(row['frame'])
    if current:
        segments.append(current)
    expected_sequence = [state for _ in range(6) for state in TURN_STATES]
    sequence_ok = [s['state'] for s in segments] == expected_sequence
    if sequence_ok:
        for index, segment in enumerate(segments):
            segment['task'] = ('RIGHT' if index < 6 else 'LEFT') + (
                '_ENTER' if segment['state'] == 'TURN_LEFT' else '_RETURN')
    outside = sum(row['state'] not in TURN_STATES for row in owned)
    linear = sum(abs(row['command'][0]) > 1e-8 for row in owned)
    return dict(turn_segments=segments,
                all_twelve_turns_bc_owned=sequence_ok and all(s['bc_frames'] > 0 for s in segments),
                bc_command_frames=len(owned), nonzero_turn_frames_missing_bc_owner=missing,
                bc_frames_outside_turn=outside, bc_frames_with_linear_motion=linear)


def audit(run, baseline=None):
    result = json.loads((run/'result.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    geometry_files = list((run/'geometry').glob('*/decisions.jsonl.gz'))
    if len(geometry_files) != 1:
        raise ValueError('Expected one geometry log per isolated test')
    with gzip.open(geometry_files[0], 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError('Empty geometry log')
    telemetry = [json.loads(line) for line in (run/'telemetry.jsonl').open()]
    poses = [row for row in telemetry if row['kind'] == 'world_pose']
    debug = [row for row in telemetry if row['kind'] == 'patrol_debug']
    if not poses:
        raise ValueError('Missing independent world pose telemetry')
    final_window = [row['data'] for row in poses
                    if poses[-1]['wall_time'] - row['wall_time'] <= .8]
    drift = max(math.hypot(row['x']-final_window[-1]['x'],
                           row['y']-final_window[-1]['y']) for row in final_window)
    source = {name: digest(ROOT/name) == value
              for name, value in manifest['source_sha256'].items()}
    contract_ok = all(digest(ROOT/name) == value
                      for name, value in manifest['model_contract']['hashes'].items())
    model_ok = digest(Path(manifest['model'])) == manifest['model_sha256']
    log = (run/'patrol.log').read_text().splitlines()
    ownership = turn_ownership(rows)
    report = dict(
        route_passed=all(result.get(key, False) for key in (
            'fsm_done', 'independent_endpoint_checks_passed',
            'return_to_start_region', 'zero_command_confirmed')),
        **ownership,
        source_checks=source, source_unchanged=all(source.values()),
        model_unchanged=model_ok, model_contract_unchanged=contract_ok,
        wall_duration_s=result['elapsed_wall_s'],
        sim_duration_s=rows[-1]['scan_stamp']-rows[0]['scan_stamp'],
        geometry_frames=len(rows), gzip_footer_complete=True,
        states=dict(Counter(row['state'] for row in rows)),
        endpoint_checks=result['endpoint_checks'], final_world_pose=result['world_pose'],
        final_distance_from_original_start_m=math.hypot(
            result['world_pose']['x'], result['world_pose']['y']+11.3),
        final_0_8_s_planar_drift_m=drift, final_window_pose_samples=len(final_window),
        localization_recovery_frames=sum(bool(row['localization_recovery']) for row in rows),
        turn_retry_observed=any(row['data'].get('turn_reposition_attempts', 0) > 0 for row in debug),
        errors=[line for line in log if '[ERROR]' in line or 'Traceback' in line],
        warnings=[line for line in log if '[WARN]' in line],
        scan_receive_gaps_over_0_4_s=result['scan_receive_gaps_over_0_4_s'],
        max_scan_gap_wall_s=result['max_observed_scan_gap_s'],
        max_abs_roll_deg=max(abs(row['data']['roll']) for row in poses)*180/math.pi,
        max_abs_pitch_deg=max(abs(row['data']['pitch']) for row in poses)*180/math.pi,
        contacts_instrumented=False,
        scope='Fixed-terrain Gazebo repeat; no physical-contact or real-robot certification')
    report['passed'] = all((report['route_passed'], report['all_twelve_turns_bc_owned'],
        not report['nonzero_turn_frames_missing_bc_owner'], not report['bc_frames_outside_turn'],
        not report['bc_frames_with_linear_motion'], report['source_unchanged'],
        model_ok, contract_ok, not report['errors']))
    if baseline:
        previous = json.loads((baseline/'ownership_audit.json').read_text())
        old_manifest = json.loads((baseline/'manifest.json').read_text())
        report['comparison'] = dict(
            baseline=str(baseline),
            same_model=manifest['model_sha256'] == old_manifest['model_sha256'],
            same_model_contract=manifest['model_contract'] == old_manifest['model_contract'],
            same_arguments=manifest['arguments'] == old_manifest['arguments'],
            previous_route_passed=previous['route_passed'],
            wall_time_difference_s=report['wall_duration_s']-previous['wall_duration_s'],
            previous_bc_command_frames=previous['bc_command_frames'],
            previous_scan_receive_gaps=previous['scan_receive_gaps_over_0_4_s'],
            timing_note='Wall-clock variation is not a controlled performance improvement experiment')
    (run/'ownership_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.baseline)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
