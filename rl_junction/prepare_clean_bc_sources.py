#!/usr/bin/env python3
"""Keep original evaluation episodes; exclude any fault-injected training episode."""
import argparse,json,shutil
from pathlib import Path
from phase_junction_bc import digest
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);meta=json.loads((a.source/'dataset.json').read_text());kept=[];excluded=[]
for e in meta['episodes']:
 train=e['case']['split']=='train'
 bad=bool(e.get('events') or e['case'].get('turn_disturbance_m') or e.get('actuator_inhibit_applied') or 'inhibit' in e['case']['family'])
 if train and bad:excluded.append(e['case']['seed']);continue
 kept.append(e)
 if 'file' in e:shutil.copy2(a.source/e['file'],a.output/e['file'])
meta.update(episodes=kept,plan=[e['case'] for e in kept],clean_filter=dict(source=str(a.source.resolve()),source_sha256=digest(a.source/'dataset.json'),excluded_train_seeds=excluded,note='Evaluation episodes unchanged; no training episode with a synthetic event retained'))
(a.output/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n');print('Kept',len(kept),'excluded train',len(excluded))
