#!/usr/bin/env python3
"""Replay recorded scans to test the exact old conflict; not closed-loop success."""
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'rl_junction'))
from phase_junction_bc import PhasePolicy
from phase_junction_recovery import RecoveringPhaseJunctionController
from navigation import load_config,scan_features
folder=ROOT/'logs/phase_junction_bc/20261117/heldout/02_full_left_enter'
rows=[json.loads(l) for l in gzip.open(folder/'decisions.jsonl.gz','rt')]
a=np.load(folder/'scans.npz');meta=json.loads((ROOT/'logs/rough_bc_demos/diverse_20261116_v2/left_enter_20261718.json').read_text())['scan_geometry']
c=RecoveringPhaseJunctionController(load_config(ROOT/'rl_pig_pen/patrol_config_right_then_left.json'),PhasePolicy(ROOT/'logs/phase_junction_bc/20261117/model/model.pt'))
c.pass_side='LEFT';c.main_direction=-1;c.state='MAIN';out=[]
for i,row in enumerate(rows):
 t=row['stamp'];measured=np.zeros(2) if i==0 else np.array(rows[i-1]['executed'])
 # During the original safety hold, actual world-pose differences confirm rest.
 # World pose is ONLY used to validate the test's stationary condition.
 if i>22:
  prev=rows[i-1];delta=np.array(row['world_pose_privileged'])-np.array(prev['world_pose_privileged']);dt=t-prev['stamp']
  measured=np.array([np.linalg.norm(delta[:2])/dt,delta[2]/dt])
 c.observe_motion(*measured,t)
 scan=SimpleNamespace(ranges=a['ranges'][i],**{k:v for k,v in meta.items() if k!='rays'})
 cmd=c.command(scan_features(scan,c.cfg),t);c.commit(cmd)
 out.append(dict(step=i,relocalizing=c.relocalization is not None,state=c.state,command=cmd.tolist(),diagnostics=c.diagnostics.copy()))
 if c.wall_recovery_events:break
result=dict(passed=bool(c.wall_recovery_events),events=c.wall_recovery_events,rows=out,note='Fixed recorded scan replay. Not a closed-loop navigation success; scan angles from matching simulator dataset metadata.')
p=ROOT/'logs/phase_junction_recovery/20261117/recorded_replay.json';p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if not result['passed']:raise SystemExit(1)
