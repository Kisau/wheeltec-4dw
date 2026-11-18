#!/usr/bin/env python3
"""Original actor audit plus recovery-supervisor provenance and stopped handoff."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_phase_junction_bc import evaluation,ROOT,digest
path=Path(sys.argv[1]);result=evaluation(path);meta=json.loads((path/'evaluation.json').read_text())
supervisor=path/'supervisor_snapshot.py'
if not supervisor.exists():supervisor=ROOT/'rl_pig_pen/phase_junction_recovery.py'
if digest(supervisor)!=meta['supervisor_sha256']:result['errors'].append('Recovery source changed')
snapshot=path/'evaluator_snapshot.py'
evaluator=snapshot if snapshot.exists() else ROOT/'rl_junction/evaluate_phase_recovery.py'
if digest(evaluator)!=meta['evaluator_sha256']:result['errors'].append('Evaluator source mismatch')
count=0
for e in meta['episodes']:
 rows=[json.loads(l) for l in gzip.open(path/e['folder']/'decisions.jsonl.gz','rt')]
 for event in e.get('wall_recovery_events',[]):
  matching=[r for r in rows if abs(r['stamp']-event['stamp'])<1e-5]
  if not matching or matching[0]['command']!=[0.,0.]:result['errors'].append('Recovery resumed without zero handoff')
  count+=1
result.update(wall_recoveries=count,audit_passed=not result['errors'])
(path/'recovery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
if not result['audit_passed']:raise SystemExit(1)
