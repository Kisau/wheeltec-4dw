#!/usr/bin/env python3
"""Read-only audit of paired diversified BC evaluations; no ROS required."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rl_junction'))
from rough_demo_common import Case,goal_check


def audit(path,partial=False):
    meta=json.loads((path/'evaluation.json').read_text());errors=[];results=[]
    if not partial and not meta.get('complete'):errors.append('Evaluation incomplete')
    if not meta.get('models_unchanged',partial):errors.append('Model weights changed')
    if not partial and len(meta['episodes'])!=len(meta['plan']):errors.append('Missing planned attempts')
    for index,e in enumerate(meta['episodes']):
        expected=meta['plan'][index]
        if e['case']!=expected['case'] or e['model']!=expected['model']:errors.append(f'{index}: plan changed')
        folder=path/e['folder']
        with gzip.open(folder/'decisions.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        calls=json.loads((folder/'model_calls.json').read_text());stamps=[r['stamp'] for r in rows]
        call_stamps=[c['stamp'] for c in calls]
        if len(stamps)!=len(set(stamps)) or np.any(np.diff(stamps)<=0):errors.append(f'{index}: scan stamps invalid')
        if len(call_stamps)!=len(set(call_stamps)):errors.append(f'{index}: repeated model call')
        by_stamp={r['stamp']:r for r in rows}
        for call in calls:
            row=by_stamp.get(call['stamp'])
            if row is None or row['phase'] not in ('TURN_LEFT','TURN_MAIN') or abs(row['command'][0])>1e-8:
                errors.append(f'{index}: invalid model ownership');continue
            limit=.25 if call['retry'] else .6
            # Safety may zero a model request; otherwise the controller enforces the same runtime cap.
            expected_w=float(np.clip(call['raw_angular_z'],-limit,limit))
            if abs(row['command'][1])>1e-8 and not np.isclose(row['command'][1],expected_w,atol=1e-7):
                errors.append(f'{index}: runtime action differs from bounded policy')
        for row in rows:
            pure_turn=(row['phase']==row['next_phase'] and row['phase'] in ('TURN_LEFT','TURN_MAIN'))
            if pure_turn and abs(row['command'][1])>1e-7 and row['stamp'] not in call_stamps:
                errors.append(f'{index}: nonzero turn without model')
            if row['model_called']!=(row['stamp'] in call_stamps):errors.append(f'{index}: call count mismatch')
        if e['steps']!=len(rows) or e['bc_calls']!=len(calls):errors.append(f'{index}: metadata count mismatch')
        with np.load(folder/'scans.npz',allow_pickle=False) as scans:
            if not np.array_equal(scans['stamps'],stamps):errors.append(f'{index}: raw scans misaligned')
        if e['success'] and (not e['final_stationary'] or not goal_check(Case(**e['case']),e['final_world_pose'])):
            errors.append(f'{index}: invalid successful handoff')
        if e['case']['turn_disturbance_m'] and e['success'] and not e['disturbance_applied']:
            errors.append(f'{index}: disturbance missing')
        results.append(dict(model=e['model'],task=e['case']['task'],family=e['case']['family'],
            success=e['success'],reason=e['reason'],bc_calls=len(calls),max_retries=e['max_retries'],
            relocalization_frames=e['relocalization_frames'],safety_intervention_frames=e['safety_intervention_frames'],
            inference_p95_ms=e['inference_p95_ms']))
    for name,info in meta['models'].items():
        if hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()!=info['sha256']:errors.append(name+': model hash mismatch')
    for name,value in meta['contract']['hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:errors.append('Asset changed: '+name)
    summary=dict(complete=meta.get('complete',False),audit_passed=not errors,errors=errors,episodes=results,
        by_model_family={name:{family:dict(attempts=sum(r['model']==name and r['family']==family for r in results),
            successes=sum(r['model']==name and r['family']==family and r['success'] for r in results))
            for family in ('nominal','heldout_recovery')} for name in ('new','baseline')},
        interpretation='Audit correctness is distinct from navigation success. Fixed-terrain small paired study; FSM owns recovery, model controls rotation only. No contact sensor or real-world deployment.')
    (path/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();r=audit(args.run,args.allow_partial)
    print(json.dumps({k:v for k,v in r.items() if k!='episodes'},indent=2))
    raise SystemExit(0 if r['audit_passed'] else 1)
