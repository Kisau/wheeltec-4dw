#!/usr/bin/env python3
"""Audit clean supervision, frozen BC identity, residual bounds and physics evidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'rl_junction'))
from phase_junction_bc import digest
from stable_baselines3 import PPO
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--evaluation',type=Path);a=p.parse_args();r=a.root;errors=[];valid_labels=0
for name in ('clean_base','clean_fsm'):
 meta=json.loads((r/name/'dataset.json').read_text())
 if not meta.get('complete'):errors.append(name+': incomplete')
 for e in meta['episodes']:
  if e['case']['split']!='train' or not e.get('bc_eligible'):continue
  if e.get('events') or e['case'].get('turn_disturbance_m') or e.get('actuator_inhibit_applied'):errors.append('Fault-injected demonstration used')
  data=np.load(r/name/e['file']);mask=data['label_valid'];valid_labels+=int(mask.sum())
  if (mask&(data['relocalizing']|data['obstacle'])).any():errors.append('Unsafe label')
  if digest(r/name/e['file'])!=e['sha256']:errors.append('Demonstration changed')
provenance=json.loads((r/'bc/provenance.json').read_text())
if digest(r/'bc/model.pt')!=provenance['weights_sha256'] or digest(r/'bc/model.json')!=provenance['metadata_sha256']:errors.append('BC provenance mismatch')
t=json.loads((r/'ppo/training.json').read_text())
if digest(r/'bc/model.pt')!=t['bc_sha256']:errors.append('Frozen BC changed')
for name,expected in t['sources'].items():
 if digest(r/'ppo/source_snapshot'/name)!=expected:errors.append('Training source snapshot mismatch: '+name)
model=PPO.load(r/'ppo/residual_ppo.zip',device='cpu')
actor_changed=bool(model.policy.action_net.weight.detach().abs().max()>0)
if not actor_changed:errors.append('Residual actor never updated')
transitions=0
for line in (r/'ppo/rollout/transitions.jsonl').open():
 x=json.loads(line);delta=np.asarray(x['residual']);transitions+=1
 if not np.isfinite(delta).all() or np.any(np.abs(delta)>np.array([.03,.12])+1e-6):errors.append('Residual bound violation')
eval_path=(a.evaluation/'evaluation.json') if a.evaluation else r/'paired/evaluation.json';evaluation=None
if eval_path.exists():
 m=json.loads(eval_path.read_text());folder=eval_path.parent
 for name,expected in m['sources'].items():
  if digest(folder/'source_snapshot'/name)!=expected:errors.append('Evaluation source snapshot mismatch: '+name)
 if digest(r/'bc/model.pt')!=m['bc_sha256'] or digest(r/'ppo/residual_ppo.zip')!=m['ppo_sha256']:errors.append('Evaluation checkpoint changed')
 pairs={(e['scenario'],e['mode']) for e in m['episodes']}
 if pairs!={(i,mode) for i in range(4) for mode in ('bc','residual')}:errors.append('Missing/duplicate evaluation pairs')
 if len(m['episodes'])!=len(pairs):errors.append('Duplicate episode')
 if not m['complete'] or not m['restored_original']:errors.append('Evaluation incomplete/not restored')
 evaluation={mode:dict(attempts=sum(x['mode']==mode for x in m['episodes']),successes=sum(x['mode']==mode and x['success'] for x in m['episodes'])) for mode in ('bc','residual')}
 for e in m['episodes']:
  if e['success'] and not e['final_stationary']:errors.append('Success without stationary confirmation')
  if e['case']!=m['plan'][e['scenario']]['case']:errors.append('Paired starting case mismatch')
  rows=[json.loads(line) for line in (folder/f"transitions_{e['scenario']}_{e['mode']}.jsonl").open()]
  target='LEFT_OUTBOUND' if e['case']['task'].endswith('ENTER') else 'MAIN'
  if any(x.get('next_phase')==target for x in rows[:-1]):errors.append('Continued after an invalid handoff')
  if len(rows)!=e['steps']:errors.append('Step count mismatch')
  if sum(x['fsm_correction'] for x in rows)!=e['fsm_correction_frames'] or sum(x['projection'] for x in rows)!=e['projection_frames']:errors.append('Intervention count mismatch')
  for x in rows:
   delta=np.asarray(x['residual'])
   if not np.isfinite(delta).all() or np.any(np.abs(delta)>np.array([.03,.12])+1e-6):errors.append('Evaluation residual bound violation')
   if (not x['base_owned'] or e['mode']=='bc') and np.any(delta!=0):errors.append('Residual applied outside ownership or in baseline')
   if x['phase'].startswith('TURN') and delta[0]!=0:errors.append('Translation residual during rotation')
   if x['owner']=='SAFETY_HOLD' and np.any(np.asarray(x['executed'])!=0):errors.append('Safety hold moved')
  live=e['dynamics']['live_verification']
  if digest(live['live_sdf'])!=live['sha256']:errors.append('Physical evidence changed')
  if abs(live['total_mass_kg']-(21.55+e['dynamics']['payload_kg']))>.05:errors.append('Payload mismatch')
  if len(live['wheel_mu'])!=4 or len(live.get('wheel_mu2',[]))!=4:errors.append('Incomplete wheel friction evidence')
  if any(abs(mu-e['dynamics']['friction'])>.001 for mu in live['wheel_mu']+live.get('wheel_mu2',[])):errors.append('Friction mismatch')
result=dict(passed=not errors,errors=errors,valid_train_labels=valid_labels,ppo_steps=t['steps'],logged_transitions=transitions,residual_actor_updated=actor_changed,bc_unchanged=True if digest(r/'bc/model.pt')==t['bc_sha256'] else False,evaluation=evaluation)
(r/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if errors:raise SystemExit(1)
