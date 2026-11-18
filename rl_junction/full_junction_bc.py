#!/usr/bin/env python3
"""Independent full-local-junction BC: causal history, bounded linear/angular action."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from rough_demo_common import ROOT, OBS_SIZE, TASKS, contract as demo_contract
from validate_rough_demos import validate

LOCAL_PHASES=('CENTER','RETURN_CENTER','TURN_LEFT','TURN_MAIN','ENTRY_LEFT','EXIT_MAIN')
HISTORY=8
SCALE=np.array([.4,.6],dtype=np.float32)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    result=demo_contract()
    result.update(schema='full_local_junction_bc_v1',history=HISTORY,
        takeover='CENTER/RETURN_CENTER through entry/exit; straight and safety remain FSM',
        architecture='MLP 8*159 -> 128 -> 128 -> 2 tanh; train-only mean/std')
    for name in ('rl_junction/full_junction_bc.py','rl_pig_pen/full_junction_controller.py'):
        result['hashes'][name]=digest(ROOT/name)
    return result


def history_windows(x,history=HISTORY):
    """Past/current observations only, repeat first observation for left padding."""
    indices=np.maximum(np.arange(len(x))[:,None]-np.arange(history-1,-1,-1)[None,:],0)
    return np.asarray(x,dtype=np.float32)[indices]


def eligible_mask(data):
    return data['label_valid'] & np.isin(data['phases'],LOCAL_PHASES) & (data['phases']==data['next_phases'])


def load_sources(sources):
    groups={s:[] for s in ('train','validation','test')};provenance=[];seen=set()
    for index,source in enumerate(sources):
        check=validate(source,allow_partial=index>0)
        meta=json.loads((source/'dataset.json').read_text())
        if not check['passed'] or not meta.get('complete'):raise ValueError('Source validation failed: '+str(source))
        provenance.append(dict(path=str(source.resolve()),sha256=digest(source/'dataset.json')))
        for e in meta['episodes']:
            if not e.get('bc_eligible'):continue
            split=e['case']['split']
            if index>0 and split!='train':raise ValueError('Correction source must add training episodes only')
            key=e['case']['seed']
            if key in seen:raise ValueError('Duplicate seed across sources')
            seen.add(key)
            path=source/e['file']
            with np.load(path,allow_pickle=False) as a:
                x=a['observations'].astype(np.float32);mask=eligible_mask(a)
                if not mask.any():continue
                groups[split].append(dict(x=history_windows(x)[mask],y=a['teacher_actions'][mask].astype(np.float32),
                    all_x=x,phase=a['phases'][mask].copy(),task=e['case']['task'],seed=key,
                    file=str(path.resolve()),sha256=e['sha256']))
    for split,es in groups.items():
        if {e['task'] for e in es}!=set(TASKS):raise ValueError('Missing four-task coverage: '+split)
    return groups,provenance


def network():
    import torch
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('mean',torch.zeros(OBS_SIZE));self.register_buffer('std',torch.ones(OBS_SIZE))
            self.actor=torch.nn.Sequential(torch.nn.Linear(HISTORY*OBS_SIZE,128),torch.nn.Tanh(),
                torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,2),torch.nn.Tanh())
        def forward(self,x):
            x=torch.clamp((x-self.mean)/self.std,-10.,10.)
            return self.actor(x.flatten(start_dim=-2))
    return Actor()


class FullPolicy:
    def __init__(self,path):
        import torch
        torch.set_num_threads(1);path=Path(path)
        self.metadata=json.loads(path.with_suffix('.json').read_text())
        if self.metadata['contract']!=json.loads(json.dumps(contract())):raise ValueError('Full BC runtime contract changed')
        self.model=network();self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.model.eval()
    def __call__(self,history):
        import torch
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(HISTORY,OBS_SIZE) or not np.isfinite(x).all():raise ValueError('Invalid causal policy history')
        with torch.inference_mode():a=self.model(torch.from_numpy(x)).numpy()*SCALE
        if not np.isfinite(a).all():raise ValueError('Non-finite policy action')
        return a


def weighted_arrays(episodes):
    xs=[];ys=[];ws=[]
    task_counts=Counter(e['task'] for e in episodes)
    for e in episodes:
        counts=Counter(e['phase']);nphase=len(counts)
        xs.append(e['x']);ys.append(e['y'])
        ws.extend(1/(len(TASKS)*task_counts[e['task']]*nphase*counts[p]) for p in e['phase'])
    return np.concatenate(xs),np.concatenate(ys),np.array(ws,dtype=np.float32)


def train(sources,output,epochs=400):
    import torch
    torch.set_num_threads(1);groups,provenance=load_sources(sources)
    output.mkdir(parents=True,exist_ok=False)
    arrays={s:weighted_arrays(es) for s,es in groups.items()}
    tensors={s:tuple(torch.from_numpy(v) for v in values) for s,values in arrays.items()}
    train_raw=np.concatenate([e['all_x'] for e in groups['train']]);mean=train_raw.mean(0);std=np.maximum(train_raw.std(0),.05)
    def score(model,split):
        x,y,w=tensors[split];return (((model(x)-y)**2).mean(-1)*w).sum()/w.sum()
    candidates=[]
    for seed in (0,1,2):
        torch.manual_seed(seed);model=network();model.mean.copy_(torch.from_numpy(mean));model.std.copy_(torch.from_numpy(std))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-4)
        folder=output/f'seed_{seed}';folder.mkdir();best=float('inf');best_epoch=0
        with (folder/'metrics.jsonl').open('w') as log:
            for epoch in range(epochs):
                model.train();loss=score(model,'train');optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();model.eval()
                with torch.inference_mode():val=float(score(model,'validation'))
                if val<best:best=val;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'model.pt')
                if epoch%20==0 or epoch==epochs-1:
                    log.write(json.dumps(dict(epoch=epoch+1,train_mse=float(loss.detach()),validation_mse=val))+'\n');log.flush()
        candidates.append(dict(seed=seed,validation_mse=best,best_epoch=best_epoch,path=str(folder/'model.pt')))
        print('CANDIDATE',json.dumps(candidates[-1]),flush=True)
    chosen=min(candidates,key=lambda c:c['validation_mse']);(output/'model.pt').write_bytes(Path(chosen['path']).read_bytes())
    model=network();model.load_state_dict(torch.load(output/'model.pt',weights_only=True));model.eval();scores={}
    with torch.inference_mode():
        for split,(x,y,w) in tensors.items():
            err=(model(x)-y).abs()*torch.from_numpy(SCALE)
            scores[split]=dict(episodes=len(groups[split]),frames=len(x),linear_mae_m_s=float(err[:,0].mean()),
                angular_mae_rad_s=float(err[:,1].mean()),balanced_normalized_mse=float(score(model,split)))
    manifest={s:[{k:e[k] for k in ('task','seed','file','sha256')} for e in es] for s,es in groups.items()}
    report=dict(contract=contract(),sources=provenance,selection='Validation only; test scored once after checkpoint/seed selection',
        candidates=candidates,chosen=chosen,scores=scores,episodes=manifest,
        note='Full local junction v/w actor, no world pose or FSM phase in input. Closed-loop evaluation required; not deployed.')
    (output/'model.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(weights_sha256=digest(output/'model.pt'),metadata_sha256=digest(output/'model.json')),indent=2)+'\n')
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=400);args=p.parse_args()
    if args.epochs<1:p.error('epochs must be positive')
    train(args.source,args.output,args.epochs)
