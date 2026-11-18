#!/usr/bin/env python3
"""Mirror custom BC metrics live, preserving existing trainer/model source contracts.

Example: python3 rl_junction/train_with_tensorboard.py --script phase_junction_bc.py -- --source DATA --output NEW_RUN --epochs 400
"""
import argparse,csv,json,math,runpy,sys
from pathlib import Path
from unittest.mock import patch
from torch.utils.tensorboard import SummaryWriter

class MetricMirror:
 def __init__(self,file,path,writers):
  self.file=file;self.path=path;self.buffer='';self.header=None
  key=path.resolve()
  if key not in writers:
   writer=SummaryWriter(str(path.parent/'tensorboard'));writer.add_text('run/recording','Live BC metric mirror; original trainer unchanged.')
   writers[key]=writer
  self.writer=writers[key]
 def write(self,text):
  result=self.file.write(text);self.buffer+=text
  while '\n' in self.buffer:
   line,self.buffer=self.buffer.split('\n',1)
   if not line.strip():continue
   if self.path.suffix=='.jsonl':row=json.loads(line)
   else:
    fields=next(csv.reader([line]))
    if self.header is None:self.header=fields;continue
    row=dict(zip(self.header,fields))
   step=int(float(row.get('epoch',row.get('step',0))))
   for key,value in row.items():
    if key in ('epoch','step'):continue
    try:value=float(value)
    except (ValueError,TypeError):continue
    if math.isfinite(value):self.writer.add_scalar('bc/'+key,value,step)
  self.writer.flush();return result
 def __getattr__(self,key):return getattr(self.file,key)
 def __enter__(self):self.file.__enter__();return self
 def __exit__(self,*args):self.writer.flush();return self.file.__exit__(*args)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--script',type=Path,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
 script=a.script if a.script.is_absolute() else Path(__file__).parent/a.script
 if not script.is_file():raise FileNotFoundError(script)
 args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
 original=Path.open;writers={}
 def open_logged(path,*args,**kwargs):
  file=original(path,*args,**kwargs);mode=kwargs.get('mode',args[0] if args else 'r')
  if path.name in ('metrics.csv','metrics.jsonl') and 'w' in mode and 'b' not in mode:return MetricMirror(file,path,writers)
  return file
 old_argv=sys.argv[:];sys.argv=[str(script),*args];sys.path.insert(0,str(script.parent))
 try:
  with patch.object(Path,'open',open_logged):runpy.run_path(str(script),run_name='__main__')
 finally:
  for writer in writers.values():writer.close()
  sys.argv=old_argv;sys.path.pop(0)
if __name__=='__main__':main()
