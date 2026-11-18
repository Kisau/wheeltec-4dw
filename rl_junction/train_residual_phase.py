#!/usr/bin/env python3
"""Train or resume bounded residual PPO; BC stays frozen and TensorBoard is required."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from residual_phase_env import ResidualPhaseEnv,RESIDUAL_LIMIT
from phase_junction_bc import digest


def validate_parent(checkpoint,bc):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    metadata=json.loads(checkpoint.with_name('training.json').read_text())
    if not metadata.get('complete'):raise ValueError('Parent training is incomplete')
    if digest(bc)!=metadata['bc_sha256']:raise ValueError('Resume requires the same frozen BC')
    if not np.allclose(metadata['residual_limits'],RESIDUAL_LIMIT):raise ValueError('Residual limits changed')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bc',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=512,help='Additional steps, a multiple of 64')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume',type=Path,help='Existing residual_ppo.zip; requires adjacent training.json')
    args=parser.parse_args()
    if args.steps<64 or args.steps%64:parser.error('steps must be a positive multiple of 64')
    parent=validate_parent(args.resume,args.bc) if args.resume else None
    seed=args.seed if args.seed is not None else parent.get('seed',0) if parent else 0
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    raw=ResidualPhaseEnv(args.bc,args.output/'rollout',seed)
    model=None
    env=None
    try:
        env=Monitor(raw,str(args.output/'episodes'))
        if args.resume:
            # SB3 restores both policy/value parameters and optimizer state.
            # Physics/history restart in a fresh episode; no claim of exact trajectory replay.
            model=PPO.load(args.resume,env=env,device='cpu',seed=seed,force_reset=True)
            if model.n_steps!=64:raise ValueError('Expected 64-step rollout checkpoint')
            if model.num_timesteps!=parent['steps']:raise ValueError('Checkpoint/metadata step mismatch')
        else:
            model=PPO('MlpPolicy',env,n_steps=64,batch_size=32,n_epochs=4,
                learning_rate=1e-4,gamma=.99,gae_lambda=.95,clip_range=.1,
                ent_coef=0.,target_kl=.02,seed=seed,device='cpu',verbose=1,
                policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-2.))
            torch.nn.init.zeros_(model.policy.action_net.weight)
            torch.nn.init.zeros_(model.policy.action_net.bias)
        model.set_logger(configure(str(args.output/'metrics'),['stdout','csv','json','tensorboard']))
        starting_steps=model.num_timesteps
        initial=[p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=args.steps,reset_num_timesteps=not bool(args.resume))
        model.logger.record('training/final_dump',1)
        model.logger.dump(step=model.num_timesteps)
        model.save(args.output/'residual_ppo')
        changed=any(not torch.equal(x,y) for x,y in zip(initial,model.policy.parameters()))
        metadata=dict(tensorboard_log=str((args.output/'metrics').resolve()),
            bc_path=str(args.bc.resolve()),bc_sha256=raw.bc_hash,
            steps=model.num_timesteps,starting_steps=starting_steps,
            additional_steps=model.num_timesteps-starting_steps,
            resume=None if not args.resume else dict(path=str(args.resume.resolve()),
                sha256=digest(args.resume),mode='weights_and_optimizer_with_new_episode'),
            residual_limits=RESIDUAL_LIMIT.tolist(),residual_parameters_changed=changed,
            seed=seed,complete=True,
            sources={name:digest(Path(__file__).parent/name) for name in
                ('residual_phase_env.py','train_residual_phase.py')},
            note='BC frozen; safeguards retained. Independent validation/evaluation still required.')
        (args.output/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        try:
            if model is not None and getattr(model,'_logger',None) is not None:model.logger.close()
        finally:
            if env is not None:env.close()
            else:raw.close()


if __name__=='__main__':main()
