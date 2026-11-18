#!/usr/bin/env python3
"""Turn-only BC pilot from audited rough-world LiDAR teacher recordings.

No world pose enters the policy. Complete turns are the split unit.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'rl_pig_pen'))
from navigation import load_config, scan_features

TASKS = ('RIGHT_ENTER', 'RIGHT_RETURN', 'LEFT_ENTER', 'LEFT_RETURN')
SCHEMA = 'rough_right_left_turn_only_bc_v2'


def task_name(side, state):
    return side + ('_ENTER' if state == 'TURN_LEFT' else '_RETURN')


def encode(features, side, state, centre, heading):
    """72 angular min bins, task4, local centre2, heading sin/cos2."""
    bins = np.minimum((np.mod(features.angles+math.pi, 2*math.pi)*72/(2*math.pi)).astype(int),71)
    ranges = np.full(72, 10.)
    np.minimum.at(ranges, bins, features.ranges)
    task = np.eye(4)[TASKS.index(task_name(side, state))]
    x = np.r_[ranges/10., task, np.clip(np.asarray(centre)/.15,-2,2), math.sin(heading), math.cos(heading)]
    if x.shape != (80,) or not np.isfinite(x).all():
        raise ValueError('Invalid turn observation')
    return x.astype(np.float32)


def contract():
    paths = ['rl_junction/turn_bc.py', 'rl_pig_pen/lidar_patrol_right_then_left.py',
             'rl_pig_pen/navigation.py', 'rl_pig_pen/lidar_geometry.py',
             'rl_pig_pen/junction_localization.py', 'rl_pig_pen/patrol_config_right_then_left.json',
             'worlds/pig_pen_16units_rough.world', 'worlds/pig_pen_16units_rough.terrain.json',
             'turn_on_wheeltec_robot/urdf/four_wheel_diff_bs_robot.urdf']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'worlds/meshes/rough_pen').glob('*_gravel.stl'))]
    return dict(schema=SCHEMA, observation_size=80, tasks=TASKS,
                action='v=0; w=.6*normalized_action; FSM alignment/clearance guards retained',
                hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


def network():
    import torch
    class TurnFeedbackPolicy(torch.nn.Module):
        """Learn bounded positive feedback gain, avoiding spurious zero equilibria.

        The network determines speed; measured heading supplies direction.
        FSM still owns alignment confirmation, drift and clearance guards.
        """
        def __init__(self):
            super().__init__()
            self.gain=torch.nn.Sequential(torch.nn.Linear(80,64),torch.nn.Tanh(),
                torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
        def forward(self,x):
            heading=torch.atan2(x[..., -2:-1],x[..., -1:])
            gain=.25+3.75*torch.sigmoid(self.gain(x))
            return torch.clamp(heading*gain,-.6,.6)/.6
    return TurnFeedbackPolicy()


class TurnPolicy:
    def __init__(self, path):
        import torch
        torch.set_num_threads(1)
        path=Path(path)
        self.metadata=json.loads(path.with_suffix('.json').read_text())
        # JSON serializes task tuples as lists.
        if self.metadata['contract'] != json.loads(json.dumps(contract())):
            raise ValueError('BC contract differs: retrain explicitly, do not edit metadata')
        self.model=network();self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.model.eval()
        self.calls=0

    def __call__(self, features, controller, heading):
        import torch
        x=encode(features,controller.pass_side,controller.state,controller.tracker.geometry.centre,heading)
        with torch.inference_mode():value=float(self.model(torch.from_numpy(x)).item())*.6
        self.calls+=1
        return value


def collect(run, output):
    """Keep only successful, unblocked measured turns, excluding transition frames."""
    result=json.loads((run/'result.json').read_text())
    if not all(result[k] for k in ('fsm_done','independent_endpoint_checks_passed','return_to_start_region','zero_command_confirmed')):
        raise ValueError('Teacher route audit did not pass')
    manifest=json.loads((run/'manifest.json').read_text())
    # All geometry inputs and physical meshes must match. The optional policy
    # hook changes controller source, so preserve its original hash as provenance.
    expected=contract()['hashes']
    for name, saved in manifest['sha256'].items():
        if name in expected and name != 'rl_pig_pen/lidar_patrol_right_then_left.py' and expected[name]!=saved:
            raise ValueError('Teacher input contract changed: '+name)
    directory=next((run/'geometry').glob('scan_*'))
    with gzip.open(directory/'decisions.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
    cfg=load_config(ROOT/'rl_pig_pen/patrol_config_right_then_left.json')
    groups=[];current=None;previous=None;chunk_id=None;chunk=None
    for row in records:
        state=row['state'];key=state if state in ('TURN_LEFT','TURN_MAIN') else None
        if key!=previous:
            if current and current['x']:groups.append(current)
            current=None
            if key:
                side='RIGHT' if len(groups)<6 else 'LEFT'
                current=dict(task=task_name(side,key),x=[],y=[],stamps=[])
        previous=key
        if not current:continue
        diag=row['controller'];tracker=row['tracker']
        if (diag.get('clearance_check')=='before_turn' or row['localization_recovery'] or
                diag.get('external_pause') or not tracker or 'heading_error' not in diag):continue
        heading=diag['heading_error'];w=row['command'][1]
        if abs(row['command'][0])>1e-8 or abs(heading)<=cfg.yaw_tolerance or abs(w)<1e-7:continue
        if chunk_id!=row['raw_chunk']:
            chunk_id=row['raw_chunk']
            with np.load(directory/f'scans_{chunk_id:05d}.npz',allow_pickle=False) as data:chunk=data['ranges'].copy()
        s=SimpleNamespace(ranges=chunk[row['raw_index']],**row['scan_geometry'])
        f=scan_features(s,cfg)
        current['x'].append(encode(f,side,key,tracker['centre'],heading));current['y'].append([w/.6]);current['stamps'].append(row['scan_stamp'])
    if current and current['x']:groups.append(current)
    if len(groups)!=12:raise ValueError(f'Expected twelve complete turn segments, got {len(groups)}')
    output.mkdir(parents=True,exist_ok=True)
    entries=[];counts={task:0 for task in TASKS}
    for i,g in enumerate(groups):
        rank=counts[g['task']];counts[g['task']]+=1;split=('train','validation','test')[rank]
        file=output/f'turn_{i:02d}.npz';np.savez_compressed(file,x=np.asarray(g['x']),y=np.asarray(g['y'],dtype=np.float32),stamps=g['stamps'])
        entries.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),task=g['task'],split=split,frames=len(g['x'])))
    metadata=dict(contract=contract(),source_run=str(run.resolve()),source_manifest=manifest,
        source_decisions_sha256=hashlib.sha256((directory/'decisions.jsonl.gz').read_bytes()).hexdigest(),
        note='Pilot only: one fixed-terrain route, three turns/task, split by complete turn; not independent terrain/generalization evidence. CPU-load incident retained in source.',episodes=entries)
    (output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def train(data, output, epochs, seed):
    import torch
    torch.set_num_threads(1);torch.manual_seed(seed)
    meta=json.loads((data/'dataset.json').read_text())
    if meta['contract']!=json.loads(json.dumps(contract())):raise ValueError('Dataset contract changed')
    arrays={s:([],[]) for s in ('train','validation','test')}
    for e in meta['episodes']:
        file=data/e['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Dataset hash mismatch')
        with np.load(file,allow_pickle=False) as a:
            arrays[e['split']][0].append(a['x']);arrays[e['split']][1].append(a['y'])
    tensors={s:tuple(torch.from_numpy(np.concatenate(v)).float() for v in pair) for s,pair in arrays.items()}
    model=network();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);best=float('inf');output.mkdir(parents=True,exist_ok=True)
    with (output/'metrics.jsonl').open('w') as log:
        for epoch in range(epochs):
            model.train();x,y=tensors['train'];order=torch.randperm(len(x))
            for batch in order.split(64):
                loss=torch.nn.functional.mse_loss(model(x[batch]),y[batch]);optimizer.zero_grad();loss.backward();optimizer.step()
            model.eval()
            with torch.inference_mode():val=torch.nn.functional.mse_loss(model(tensors['validation'][0]),tensors['validation'][1]).item()
            if val<best:best=val;torch.save(model.state_dict(),output/'model.pt')
            if epoch%50==0 or epoch==epochs-1:log.write(json.dumps(dict(epoch=epoch+1,validation_mse=val))+'\n');log.flush()
    model.load_state_dict(torch.load(output/'model.pt',weights_only=True));model.eval();scores={}
    with torch.inference_mode():
        for s,(x,y) in tensors.items():scores[s]=dict(frames=len(x),angular_mae_rad_s=float((model(x)-y).abs().mean())*.6,normalized_mse=float(((model(x)-y)**2).mean()))
    report=dict(contract=contract(),dataset=str((data/'dataset.json').resolve()),seed=seed,epochs=epochs,scores=scores,scope='Offline turn-only pilot; closed-loop evaluation required')
    (output/'model.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['scores'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['collect','train']);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=1000);parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    if args.mode=='collect':print(json.dumps(collect(args.source,args.output)['episodes'],indent=2))
    else:train(args.source,args.output,args.epochs,args.seed)
