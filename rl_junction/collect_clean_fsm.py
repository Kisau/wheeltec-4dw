#!/usr/bin/env python3
"""New initial-offset FSM demonstrations, with no mid-rollout fault injection."""
import argparse,json,math,signal,threading
from pathlib import Path
from dataclasses import asdict
from collect_correction_demos import correction_episode
from collect_rough_demos import transport,save_episode
from rough_demo_common import Case,TASKS,contract
from phase_junction_bc import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cases=[Case(t,20900000+i,'train','clean_initial_offset',(-6.18,0.,6.18)[i%3],.06 if i%2==0 else -.06,math.radians(-6 if i%2==0 else 6),.3) for i,t in enumerate(TASKS)]
 spec=contract();spec['hashes']['rl_junction/collect_clean_fsm.py']=digest(__file__)
 spec['hashes']['rl_junction/collect_correction_demos.py']=digest(Path(__file__).with_name('collect_correction_demos.py'))
 meta=dict(contract=spec,plan=[asdict(c) for c in cases],episodes=[],fault_injection=False)
 def save(): (a.output/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n')
 save();stop=threading.Event()
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
 env=transport();env.progress_path=a.output/'progress.json'
 try:
  for case in cases:
   if stop.is_set():break
   print('START',case.task,flush=True);frames,r=correction_episode(env,case,300,stop)
   if r['events']:raise RuntimeError('Clean collection unexpectedly injected an event')
   r['bc_eligible']=bool(r['success'] and r['valid_labels']>=10)
   save_episode(a.output,case,frames,r);meta['episodes'].append(r);save();print('END',case.task,r['reason'],flush=True)
 finally:
  env.finish();meta['complete']=len(meta['episodes'])==len(cases) and not stop.is_set();save()
if __name__=='__main__':main()
