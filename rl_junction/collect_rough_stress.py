#!/usr/bin/env python3
"""Separate lateral-bound stress cases; never include them as BC training labels."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import threading
from rough_demo_common import TASKS, ROOT, Case, contract
from collect_rough_demos import transport, episode, save_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[Case(task,20270000+i,'stress','lateral_bound_stress',(-6.18,0.,6.18)[i%3],
                0.,0.,.2,.17,'lateral') for i,task in enumerate(TASKS)]
    spec=contract();name='rl_junction/collect_rough_stress.py'
    spec['hashes'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    metadata=dict(contract=spec,plan=[asdict(c) for c in cases],episodes=[],stress_only=True,
        note='Large lateral centre displacement tests existing guard limits; all cases excluded from BC training regardless of outcome.')
    (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stopping=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.set())
    env=None
    try:
        env=transport();env.progress_path=args.output/'progress.json'
        for case in cases:
            if stopping.is_set():break
            print('START',json.dumps(asdict(case)),flush=True)
            try:
                frames,result=episode(env,case,240,stopping)
            except Exception as exc:
                frames=getattr(env,'current_case_frames',[])
                result=dict(case=asdict(case),success=False,bc_eligible=False,stress_only=True,
                    reason='infrastructure_fault',error=str(exc),frames=len(frames),
                    valid_labels=sum(row['label_valid'] for row in frames))
                save_episode(args.output,case,frames,result);metadata['episodes'].append(result)
                metadata['infrastructure_fault']=str(exc)
                raise
            result['bc_eligible']=False;result['stress_only']=True
            save_episode(args.output,case,frames,result);metadata['episodes'].append(result)
            (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('END',json.dumps(result),flush=True)
    finally:
        if env is not None:env.finish()
        metadata['complete']=len(metadata['episodes'])==len(cases) and not stopping.is_set() and not metadata.get('infrastructure_fault')
        (args.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
