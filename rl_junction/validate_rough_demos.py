#!/usr/bin/env python3
"""Audit diversified BC rollouts without starting ROS or loading a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from rough_demo_common import ROOT, TASKS, Case, goal_check, OBS_SIZE


def validate(directory, allow_partial=False):
    meta=json.loads((directory/'dataset.json').read_text())
    errors=[];episodes=meta['episodes'];seeds=[];entries=[]
    if not allow_partial and not meta.get('complete',False):errors.append('Collection is not complete')
    for entry in episodes:
        case=Case(**entry['case']);seeds.append(case.seed)
        if 'file' not in entry:
            if entry.get('bc_eligible'):errors.append(f'{case.seed}: eligible without data')
            entries.append(dict(case=entry['case'],success=False,bc_eligible=False,reason=entry['reason']))
            continue
        path=directory/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            errors.append(f'{case.seed}: hash mismatch');continue
        with np.load(path,allow_pickle=False) as data:
            x,y,execute,mask=[data[k] for k in ('observations','teacher_actions','executed_actions','label_valid')]
            n=len(x)
            if x.shape!=(n,OBS_SIZE) or y.shape!=(n,2) or execute.shape!=(n,2) or mask.shape!=(n,):
                errors.append(f'{case.seed}: invalid observation/action/mask shape');continue
            if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(execute).all():
                errors.append(f'{case.seed}: nonfinite policy inputs or labels')
            if np.max(np.abs(y),initial=0)>1.000001 or np.max(np.abs(execute),initial=0)>1.000001:
                errors.append(f'{case.seed}: action mapping exceeded')
            if n!=entry['frames'] or int(mask.sum())!=entry['valid_labels']:
                errors.append(f'{case.seed}: metadata count mismatch')
            stamps=data['sim_stamps'];clocks=data['sim_control_stamps']
            if np.any(np.diff(stamps)<=0) or np.any(np.diff(clocks)<=0):
                errors.append(f'{case.seed}: nonmonotonic episode clocks')
            if n>1 and not np.allclose(x[1:,158],np.diff(stamps)/.2,atol=1e-4):
                errors.append(f'{case.seed}: dt input differs from measured scan interval')
            normal=np.isclose(np.diff(clocks),.2,atol=.002)
            if n>1 and not np.allclose(x[1:,156:158][normal],execute[:-1][normal],atol=1e-5):
                errors.append(f'{case.seed}: previous action is not causal')
            code=np.eye(4)[TASKS.index(case.task)]
            if not np.allclose(x[:,144:148],code):errors.append(f'{case.seed}: task code mismatch')
            if np.any(mask & (data['relocalizing']|data['obstacle'])) or np.any(x[mask,153]<.5):
                errors.append(f'{case.seed}: unsafe/stale frames used as labels')
            if entry.get('bc_eligible') and (not entry['success'] or not entry.get('final_stationary',False)
                    or not goal_check(case,entry['final_world_pose']) or int(mask.sum())<10):
                errors.append(f'{case.seed}: accepted without independent successful handoff/stop')
            if data['world_poses_privileged'].shape!=(n,3) or data['raw_ranges'].shape[0]!=n:
                errors.append(f'{case.seed}: missing privileged/raw audit arrays')
            entries.append(dict(case=entry['case'],file=str(path.resolve()),sha256=entry['sha256'],
                success=entry['success'],bc_eligible=entry['bc_eligible'],reason=entry['reason'],
                frames=n,valid_labels=int(mask.sum()),
                relocalizing_frames=int(data['relocalizing'].sum()),
                maximum_turn_retries=int(np.max(data['turn_retries'],initial=0)),
                disturbance_applied=entry.get('disturbance_applied',False)))
    if len(seeds)!=len(set(seeds)):errors.append('Duplicate episode seed; split leakage risk')
    source_current={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
                    for name,value in meta['contract']['hashes'].items()}
    if not all(source_current.values()):errors.append('Collection source/assets changed since manifest')
    baseline=meta.get('preserved_baseline_model')
    baseline_unchanged=None
    if baseline:
        model_path=Path(baseline['path'])
        baseline_unchanged=model_path.is_file() and hashlib.sha256(model_path.read_bytes()).hexdigest()==baseline['sha256']
        if not baseline_unchanged:errors.append('Preserved baseline model changed during collection')
    splits={split:[e for e in entries if e.get('bc_eligible') and e['case']['split']==split]
            for split in ('train','validation','test')}
    for split,items in splits.items():
        coverage=items if split=='train' else [e for e in entries if e['case']['split']==split]
        if not allow_partial and {e['case']['task'] for e in coverage}!=set(TASKS):
            errors.append(f'{split}: missing task coverage in '+('accepted demonstrations' if split=='train' else 'evaluation attempts'))
    summary=dict(complete=meta.get('complete',False),attempts=len(episodes),
        accepted=sum(e.get('bc_eligible',False) for e in entries),
        frames=sum(e.get('frames',0) for e in entries),
        accepted_valid_labels=sum(e.get('valid_labels',0) for e in entries if e.get('bc_eligible')),
        by_task={task:dict(attempts=sum(e['case']['task']==task for e in entries),
            accepted=sum(e['case']['task']==task and e.get('bc_eligible',False) for e in entries)) for task in TASKS},
        accepted_by_split={s:len(v) for s,v in splits.items()},
        evaluation_outcomes={s:dict(attempts=sum(e['case']['split']==s for e in entries),
            successes=sum(e['case']['split']==s and e.get('success',False) for e in entries))
            for s in ('validation','test')},
        accepted_by_family=dict(Counter(e['case']['family'] for e in entries if e.get('bc_eligible'))),
        failure_reasons=dict(Counter(e['reason'] for e in entries if not e.get('success'))),
        successful_but_excluded=sum(e.get('success',False) and not e.get('bc_eligible',False) for e in entries),
        actual_recovery_episodes=sum(e.get('bc_eligible',False) and e.get('maximum_turn_retries',0)>0 for e in entries),
        source_checks=source_current,baseline_model_unchanged=baseline_unchanged,errors=errors,passed=not errors,
        limitations=['Fixed physical gravel mesh and friction; no mass/friction randomization',
                     'Pose shifts are synthetic disturbances, not measured real slip',
                     'No contact sensor; world pose/stop checks do not certify zero contacts'],
        episodes=entries)
    (directory/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    index=dict(schema=meta['contract']['schema'],contract=meta['contract'],bc_splits=splits,
        evaluation_cases=[e for e in entries if e['case']['split']!='train'],
        note='BC splits contain only accepted complete episodes. Evaluation cases retain failed held-out attempts; do not calculate success rate on accepted episodes only.')
    (directory/'split_index.json').write_text(json.dumps(index,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    result=validate(args.directory,args.allow_partial)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','source_checks')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
