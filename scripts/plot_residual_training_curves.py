#!/usr/bin/env python3
"""Plot exported JSON with a standard Matplotlib environment (no training imports)."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=json.loads((a.directory/'data.json').read_text());records=d['updates'];episodes=d['complete_episodes']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,axes=plt.subplots(2,3,figsize=(14,8.3),constrained_layout=True)
ax=axes.flat[0];x=[e['total_steps'] for e in episodes];y=[e['reward'] for e in episodes]
ax.plot(x,y,'o-',color='#2563eb');ax.set_title('Completed episode reward (only 3 episodes)');ax.set_ylabel('Episode reward');ax.set_ylim(33,37)
for xx,yy,name in zip(x,y,('Right enter','Right return','Left enter')):ax.annotate(name,(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
items=[('train/policy_gradient_loss','Policy gradient loss','#7c3aed'),('train/value_loss','Value loss','#d97706'),('train/explained_variance','Explained variance','#059669'),('train/approx_kl','Approximate KL','#2563eb'),('train/clip_fraction','Clipped update fraction','#dc2626')]
for ax,(key,title,color) in zip(list(axes.flat)[1:],items):
 data=[r for r in records if key in r];ax.plot([r['time/total_timesteps']-d['rollout_size'] for r in data],[r[key] for r in data],'o-',color=color,markersize=4);ax.set_title(title)
 if key=='train/explained_variance':ax.axhline(0,color='#444',linestyle='--',linewidth=.8)
 if key=='train/clip_fraction':ax.yaxis.set_major_formatter(PercentFormatter(1.))
for ax in axes.flat:ax.set_xlabel('Environment steps used');ax.set_xlim(0,540)
fig.suptitle('Frozen BC + residual PPO: first 512-step training run\nHistorical log import; final optimizer batch metrics were not printed',fontsize=14)
fig.savefig(a.directory/'ppo_training.png',dpi=160);fig.savefig(a.directory/'ppo_training.svg');plt.close(fig)
print(a.directory/'ppo_training.png')
