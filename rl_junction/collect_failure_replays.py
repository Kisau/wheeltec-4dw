#!/usr/bin/env python3
"""Replay measured model-visited poses under the original teacher, retaining outcomes.

These are new teacher rollouts after a fresh reset, not fabricated successful
labels for the failed policy trajectory. World pose is reset/scoring data only.
"""
import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import signal
import threading
from collect_rough_demos import transport,save_episode
from collect_correction_demos import correction_episode
from rough_demo_common import ROOT,Case,contract
from navigation import wrap_angle
from full_junction_bc import digest


class ReplayTransport:
    def __init__(self,base,spawns):
        object.__setattr__(self,'base',base);object.__setattr__(self,'spawns',spawns)
    def __getattr__(self,key):return getattr(self.base,key)
    def __setattr__(self,key,value):setattr(self.base,key,value)
    def reset_case(self,case):
        self.base.reset_case(case);x,y,yaw,z=self.spawns[case.seed]
        self.base.teleport(x,y,yaw,z);self.base._drain()
        for _ in range(5):self.base._advance(.2)
        if abs(self.base.pose[0]-x)>.05 or abs(self.base.pose[1]-y)>.05 or abs(wrap_angle(self.base.pose[2]-yaw))>.06:
            raise RuntimeError('Replay pose did not settle')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    original=json.loads((args.source/'evaluation.json').read_text());planned=[];spawns={};references=[]
    for i,e in enumerate(original['episodes']):
        if e['success'] or e['mode']!='full':continue
        with gzip.open(args.source/e['folder']/'decisions.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
        entering=e['case']['task'].endswith('ENTER');right=e['case']['task'].startswith('RIGHT')
        yaw=math.pi/2 if entering else 0. if right else math.pi
        eligible=[r for r in rows if r['owner']=='FULL_BC' and r['phase'] in ('CENTER','RETURN_CENTER')
            and abs(wrap_angle(r['world_pose_privileged'][2]-yaw))<=.35 and not r['blocked']
            and not r['relocalizing'] and not r['obstacle']]
        if not eligible:raise ValueError('No safely replayable pose for '+e['case']['task'])
        row=max(eligible,key=lambda r:abs(wrap_angle(r['world_pose_privileged'][2]-yaw)))
        c=Case(e['case']['task'],20500000+i,'train','on_policy_teacher_replay',e['case']['junction_y'],0.,0.,.2)
        spawn=(*row['world_pose_privileged'],.11 if entering else .05)
        planned.append(c);spawns[c.seed]=spawn
        references.append(dict(seed=c.seed,source_folder=e['folder'],source_step=row['step'],spawn=list(spawn),
            decision_sha256=digest(args.source/e['folder']/'decisions.jsonl.gz')))
    if not planned:raise ValueError('No failed episodes to replay')
    args.output.mkdir(parents=True,exist_ok=False);spec=contract()
    for name in ('rl_junction/collect_failure_replays.py','rl_junction/collect_correction_demos.py'):
        spec['hashes'][name]=digest(ROOT/name)
    meta=dict(contract=spec,source_failure_run=str(args.source.resolve()),
        source_evaluation_sha256=digest(args.source/'evaluation.json'),reset_overrides=references,
        plan=[asdict(c) for c in planned],episodes=[],
        note='Fresh teacher rollouts from failed-policy visited physical poses; failed policy histories remain excluded from successful BC episodes.')
    path=args.output/'dataset.json'
    def persist():path.write_text(json.dumps(meta,indent=2)+'\n')
    persist();stop=threading.Event();env=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        env=ReplayTransport(transport(),spawns);env.progress_path=args.output/'progress.json'
        for c in planned:
            if stop.is_set():break
            print('START',c.task,spawns[c.seed],flush=True)
            try:frames,r=correction_episode(env,c,300,stop)
            except Exception as exc:meta['infrastructure_fault']=repr(exc);persist();raise
            r['reset_override']=list(spawns[c.seed]);r['bc_eligible']=bool(r['success'] and r['valid_labels']>=10)
            save_episode(args.output,c,frames,r);meta['episodes'].append(r);persist();print('END',c.task,r['reason'],flush=True)
    finally:
        if env is not None:env.finish()
        meta['complete']=len(meta['episodes'])==len(planned) and not stop.is_set() and not meta.get('infrastructure_fault');persist()


if __name__=='__main__':main()
