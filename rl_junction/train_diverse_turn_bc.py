#!/usr/bin/env python3
"""Retrain the existing guarded turn actor from diverse full-junction episodes."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rough_demo_common import ROOT, TASKS
from validate_rough_demos import validate
from turn_bc import contract, encode, network
from navigation import load_config, scan_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def turn_mask(arrays, yaw_tolerance):
    heading=np.arctan2(arrays['observations'][:,150],arrays['observations'][:,151])
    return (arrays['label_valid'] & np.isin(arrays['phases'],('TURN_LEFT','TURN_MAIN'))
        & (arrays['phases']==arrays['next_phases']) & (np.abs(heading)>yaw_tolerance)
        & (np.abs(arrays['teacher_actions'][:,0])<1e-8)
        & (arrays['teacher_actions'][:,1]*heading>0))


def prepare(source, output):
    audit=validate(source)
    if not audit['passed']:raise ValueError(audit['errors'])
    meta=json.loads((source/'dataset.json').read_text())
    cfg=load_config(ROOT/'rl_pig_pen/patrol_config_right_then_left.json')
    output.mkdir(parents=True,exist_ok=False);entries=[]
    for e in meta['episodes']:
        if not e.get('bc_eligible'):continue
        with np.load(source/e['file'],allow_pickle=False) as a:
            indices=np.flatnonzero(turn_mask(a,cfg.yaw_tolerance));xs=[]
            for i in indices:
                scan=SimpleNamespace(ranges=a['raw_ranges'][i],**e['scan_geometry'])
                features=scan_features(scan,cfg);obs=a['observations'][i]
                xs.append(encode(features,e['case']['task'].split('_')[0],str(a['phases'][i]),
                    obs[148:150]*1.5,float(np.arctan2(obs[150],obs[151]))))
            if not xs:raise ValueError('Accepted episode has no valid turn frames: '+e['file'])
            target=output/e['file']
            np.savez_compressed(target,x=np.asarray(xs),y=a['teacher_actions'][indices,1:2].astype(np.float32),
                cap=np.where(a['turn_retries'][indices]>0,.25/.6,1.).astype(np.float32)[:,None],
                source_indices=indices,stamps=a['sim_stamps'][indices])
        entries.append(dict(file=target.name,sha256=digest(target),task=e['case']['task'],
            split=e['case']['split'],frames=len(xs),source_file=e['file'],source_sha256=e['sha256'],
            source_case=e['case']))
    result=dict(contract=contract(),episodes=entries,source=str(source.resolve()),
        source_manifest_sha256=digest(source/'dataset.json'),
        extraction_sha256=digest(__file__),
        note='Original whole-episode split retained. Only successful valid nonzero pure turn labels; raw scans re-encoded with the existing 80-D runtime. Retry angular caps are applied in loss, as in deployment.')
    (output/'dataset.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def load_arrays(data):
    meta=json.loads((data/'dataset.json').read_text());groups={s:[] for s in ('train','validation','test')}
    if meta['contract']!=json.loads(json.dumps(contract())):raise ValueError('Runtime contract changed')
    for e in meta['episodes']:
        path=data/e['file']
        if digest(path)!=e['sha256']:raise ValueError('Data hash mismatch')
        with np.load(path,allow_pickle=False) as a:
            groups[e['split']].append(dict(x=a['x'].copy(),y=a['y'].copy(),cap=a['cap'].copy(),task=e['task']))
    for split,episodes in groups.items():
        if {e['task'] for e in episodes}!=set(TASKS):raise ValueError('Missing task in '+split)
    return meta,groups


def train(data, output, epochs=600, seeds=(0,1,2)):
    import torch
    torch.set_num_threads(1)
    meta,groups=load_arrays(data);output.mkdir(parents=True,exist_ok=False)
    tensors={s:[{**e,**{k:torch.from_numpy(e[k]).float() for k in ('x','y','cap')}} for e in es]
             for s,es in groups.items()}
    def loss_for(model,es):
        # Equal task, then equal whole-episode weight; long recovery turns cannot dominate.
        return torch.stack([torch.stack([((torch.clamp(model(e['x']),-e['cap'],e['cap'])-e['y'])**2).mean()
            for e in es if e['task']==task]).mean() for task in TASKS]).mean()
    candidates=[]
    for seed in seeds:
        torch.manual_seed(seed);model=network();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        best=float('inf');best_epoch=0;folder=output/f'seed_{seed}';folder.mkdir()
        with (folder/'metrics.jsonl').open('w') as log:
            for epoch in range(epochs):
                model.train();loss=loss_for(model,tensors['train']);opt.zero_grad();loss.backward();opt.step()
                model.eval()
                with torch.inference_mode():val=float(loss_for(model,tensors['validation']))
                if val<best:
                    best=val;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'model.pt')
                if epoch%50==0 or epoch==epochs-1:
                    log.write(json.dumps(dict(epoch=epoch+1,train_mse=float(loss.detach()),validation_mse=val))+'\n');log.flush()
        candidates.append(dict(seed=seed,validation_mse=best,epoch=best_epoch,path=str(folder/'model.pt')))
        print('CANDIDATE',json.dumps(candidates[-1]),flush=True)
    chosen=min(candidates,key=lambda r:r['validation_mse'])
    (output/'model.pt').write_bytes(Path(chosen['path']).read_bytes())
    model=network();model.load_state_dict(torch.load(output/'model.pt',weights_only=True));model.eval()
    baseline=ROOT/'logs/turn_bc/20261116_pilot/model_v2/model.pt'
    old=network();old.load_state_dict(torch.load(baseline,map_location='cpu',weights_only=True));old.eval()
    def scores(actor):
        result={}
        with torch.inference_mode():
            for split,es in tensors.items():
                errors=[(torch.clamp(actor(e['x']),-e['cap'],e['cap'])-e['y']).abs()*.6 for e in es]
                result[split]=dict(episodes=len(es),frames=sum(len(e['x']) for e in es),
                    angular_mae_rad_s=float(torch.cat(errors).mean()),
                    by_task={t:float(torch.cat([err for e,err in zip(es,errors) if e['task']==t]).mean()) for t in TASKS})
        return result
    report=dict(contract=contract(),dataset=str((data/'dataset.json').resolve()),dataset_sha256=digest(data/'dataset.json'),
        seed=chosen['seed'],epochs=epochs,best_epoch=chosen['epoch'],candidates=candidates,
        selection='Minimum validation task/episode-balanced MSE only; test not used for checkpoint or seed choice',
        scores=scores(model),baseline=dict(path=str(baseline),sha256=digest(baseline),scores=scores(old)),
        scope='Offline guarded turn-only BC. FSM still owns straight, centring, recovery and handoff; no claim of learning full recovery.',
        trainer_sha256=digest(__file__))
    (output/'model.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(model_sha256=digest(output/'model.pt'),
        metadata_sha256=digest(output/'model.json'),source_manifest_sha256=meta['source_manifest_sha256']),indent=2)+'\n')
    print(json.dumps(dict(selected_seed=chosen['seed'],new=report['scores'],baseline=report['baseline']['scores']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','train'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=600);args=p.parse_args()
    if args.epochs<1:p.error('epochs must be positive')
    if args.mode=='prepare':prepare(args.source,args.output)
    else:train(args.source,args.output,args.epochs)
