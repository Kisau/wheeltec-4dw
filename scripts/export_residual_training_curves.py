#!/usr/bin/env python3
"""Export actual console/Monitor data; label TensorBoard backfill explicitly."""
import argparse,csv,json,re,zipfile
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
out=root/'training_curves';out.mkdir(parents=True,exist_ok=False)
records=[];record={};section=''
for line in (root/'ppo.console.log').read_text().splitlines():
 if re.match(r'^-{5,}$',line):
  if record:records.append(record);record={}
  section='';continue
 m=re.match(r'^\|\s*([a-z_]+/)\s*\|',line)
 if m:section=m.group(1);continue
 m=re.match(r'^\|\s*([a-z_]+)\s*\|\s*([-+.0-9eE]+)\s*\|',line)
 if m:record[section+m.group(1)]=float(m.group(2))
if record:records.append(record)
with (root/'ppo/episodes.monitor.csv').open() as f:episodes=list(csv.DictReader(line for line in f if not line.startswith('#')))
steps=0
for i,e in enumerate(episodes):
 steps+=int(e['l']);e.update(episode=i+1,total_steps=steps,reward=float(e['r']),length=int(e['l']))
fields=sorted({k for r in records for k in r})
with (out/'updates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
with (out/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['episode','total_steps','reward','length','t']);w.writeheader();w.writerows({k:e[k] for k in w.fieldnames} for e in episodes)
summary=dict(total_steps=512,complete_episodes=episodes,rollout_size=64,updates=records,
 note='Historical console import. SB3 train metrics are logged before the next update: train tags use total_timesteps-64; final 512-step optimizer update metrics were not printed. No missing values fabricated.')
(out/'data.json').write_text(json.dumps(summary,indent=2)+'\n')
run=root/'tensorboard/historical_import/ppo_512'
with SummaryWriter(str(run)) as writer:
 writer.add_text('provenance/recording_mode',summary['note'])
 for r in records:
  step=int(r['time/total_timesteps'])
  for key,value in r.items():
   if key=='time/total_timesteps':continue
   writer.add_scalar(key,value,step-64 if key.startswith('train/') else step)
 for e in episodes:
  writer.add_scalar('episode/reward',e['reward'],e['total_steps']);writer.add_scalar('episode/length',e['length'],e['total_steps'])
 with zipfile.ZipFile(root/'ppo/residual_ppo.zip') as z:checkpoint=json.loads(z.read('data'))
 if isinstance(checkpoint.get('_n_updates'),int):writer.add_scalar('checkpoint/optimizer_epoch_updates',checkpoint['_n_updates'],512)
writer=None
(out/'README.md').write_text('''# PPO 訓練資料

來源為原始 console、Monitor CSV 與 checkpoint；TensorBoard 的 historical_import 是事後匯入。

512 步只有三個完整回合，第四回合未完成。reward 有起伏，不足以判定收斂。train/* 依它們實際使用的資料步數（console total_timesteps−64）繪圖，最後 512 步的更新指標原日誌未印出，沒有補造。

updates.csv 保留原 console 的步數欄位；episodes.csv 為完整回合。data.json 供繪圖使用。
''')
print(out);print(run)
