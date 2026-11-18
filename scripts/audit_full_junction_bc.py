#!/usr/bin/env python3
"""Offline audit for correction datasets and independent full-junction rollouts."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rl_junction'))
from full_junction_bc import FullPolicy,LOCAL_PHASES,HISTORY,digest
from rough_demo_common import Case,goal_check
from validate_rough_demos import validate


def corrections(path,partial=False):
    result=validate(path,allow_partial=True);meta=json.loads((path/'dataset.json').read_text())
    errors=list(result['errors']);events=0
    if not partial and not meta.get('complete'):errors.append('Correction batch incomplete')
    for entry in meta['episodes']:
        if 'file' not in entry:continue
        with np.load(path/entry['file'],allow_pickle=False) as a:
            for event in entry.get('events',[]):
                if event['type']!='synthetic_actuator_inhibit':continue
                events+=1;start=event['step'];end=min(start+event['duration_steps'],len(a['label_valid']))
                if a['label_valid'][start:end].any():errors.append(entry['file']+': inhibited label used for supervision')
                if np.any(a['executed_actions'][start:end]!=0):errors.append(entry['file']+': inhibited actuator moved')
        if entry['reason']=='no_motion_watchdog' and entry.get('bc_eligible'):errors.append('Failure accepted as BC')
    result.update(inhibit_events=events,errors=errors,passed=not errors)
    (path/'correction_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def evaluation(path,partial=False):
    meta=json.loads((path/'evaluation.json').read_text());errors=[];out=[]
    if not partial and not meta.get('complete'):errors.append('Evaluation incomplete')
    for name,info in meta['models'].items():
        if digest(info['path'])!=info['sha256']:errors.append(name+': model changed')
    for name,value in meta['contract']['hashes'].items():
        if digest(ROOT/name)!=value:errors.append('Source/assets changed: '+name)
    actor=FullPolicy(meta['models']['full']['path'])
    for index,e in enumerate(meta['episodes']):
        planned=meta['plan'][index]
        if e['mode']!=planned['mode'] or e['case']!=planned['case']:errors.append(f'{index}: plan changed')
        folder=path/e['folder']
        with gzip.open(folder/'decisions.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        stamps=np.array([r['stamp'] for r in rows]);owned=[r for r in rows if r['owner']=='FULL_BC']
        if np.any(np.diff(stamps)<=0):errors.append(f'{index}: nonmonotonic scan stamps')
        with np.load(folder/'scans.npz',allow_pickle=False) as a:
            if not np.array_equal(a['stamps'],stamps):errors.append(f'{index}: raw scans misaligned')
        with np.load(folder/'policy_inputs.npz',allow_pickle=False) as a:
            if len(owned)!=len(a['history']) or len(owned)!=e['full_calls']:errors.append(f'{index}: ownership count mismatch')
            if not np.array_equal(a['stamps'],[r['stamp'] for r in owned]):errors.append(f'{index}: actor inputs misaligned')
            for row,history in zip(owned,a['history']):
                if history.shape!=(HISTORY,159) or not np.isfinite(history).all():errors.append(f'{index}: invalid actor input');continue
                if row['phase'] not in LOCAL_PHASES or row['phase']!=row['next_phase']:errors.append(f'{index}: actor outside junction')
                if not np.allclose(actor(history),row['model_raw'],atol=1e-6):errors.append(f'{index}: recorded action is not model output')
                if history[-1,153]<.5 or row['relocalizing']:errors.append(f'{index}: actor ran without geometry')
                if abs(row['executed'][0])>.400001 or abs(row['executed'][1])>.600001:errors.append(f'{index}: bounds exceeded')
        if e['success'] and (not e['final_stationary'] or not goal_check(Case(**e['case']),e['final_world_pose'])):
            errors.append(f'{index}: false successful handoff')
        if e['success'] and e['mode']=='full' and not (e['full_centre_frames'] and e['full_turn_frames'] and e['full_exit_frames']):
            errors.append(f'{index}: success without full phase ownership')
        if e['success'] and e['case']['turn_disturbance_m'] and not e['disturbance_applied']:errors.append(f'{index}: disturbance missing')
        out.append({k:e[k] for k in ('mode','case','success','reason','steps','full_calls','full_centre_frames','full_turn_frames','full_exit_frames','full_linear_frames','safety_frames')})
    result=dict(complete=meta.get('complete',False),audit_passed=not errors,errors=errors,episodes=out,
        model_outcomes={mode:dict(attempts=sum(e['mode']==mode for e in out),successes=sum(e['mode']==mode and e['success'] for e in out)) for mode in ('full','turn')},
        note='Correct audit does not mean navigation succeeded. No contact sensors; fixed physical gravel and synthetic disturbances only.')
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('corrections','evaluation'))
    p.add_argument('directory',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    r=(corrections if a.mode=='corrections' else evaluation)(a.directory,a.allow_partial)
    print(json.dumps({k:v for k,v in r.items() if k not in ('episodes','source_checks')},indent=2))
    raise SystemExit(0 if r.get('audit_passed',r.get('passed',False)) else 1)
