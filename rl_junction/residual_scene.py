"""Own only experiment servers; embed robot at startup to avoid GPU respawn bug."""
import json,os,signal,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from residual_dynamics import robot_variant
from phase_junction_bc import digest

class ExperimentScene:
 def __init__(self,root):
  self.root=Path(root).resolve();self.state=self.root/'server_state.json'
  self.world=self.root/'scene/training.world';self.robot=self.root/'scene/training.urdf'
 def stop(self):
  if not self.state.exists():return
  data=json.loads(self.state.read_text());pid=data['pid'];proc=Path(f'/proc/{pid}/cmdline')
  if proc.exists():
   cmd=proc.read_bytes().replace(b'\0',b' ').decode()
   if data['world'] not in cmd or 'gazebo' not in cmd:raise RuntimeError('Refusing to stop unrelated PID')
   os.kill(pid,signal.SIGTERM)
   for _ in range(30):
    if not proc.exists():break
    time.sleep(.1)
   if proc.exists():os.kill(pid,signal.SIGKILL)
  self.state.unlink()
 def start(self,name,friction=1.,payload_kg=0.):
  self.stop();folder=self.root/'scenes'/name;folder.mkdir(parents=True,exist_ok=False)
  variant=robot_variant(self.robot,folder/'robot.urdf',friction,payload_kg)
  converted=subprocess.run(['ign','sdf','-p',variant['path']],capture_output=True,text=True,check=True)
  model=ET.fromstring(converted.stdout).find('model');model.set('name','wheeltec_mini')
  pose=ET.SubElement(model,'pose');pose.text='0 -11.3 0.17 0 0 1.5707963267948966'
  tree=ET.parse(self.world);tree.find('world').append(model);world=folder/'world.sdf';tree.write(world,encoding='utf-8',xml_declaration=True)
  env=os.environ.copy();env.update(DISPLAY=':1',XAUTHORITY='/run/user/1000/gdm/Xauthority',IGN_PARTITION='4wd_env_47',GZ_PARTITION='4wd_env_47')
  resources=str(self.root.parents[2]/'worlds')+':'+str(self.root.parents[2]/'pig_model/models')+':'+str(self.root.parents[2])
  env['IGN_GAZEBO_RESOURCE_PATH']=resources;env['GZ_SIM_RESOURCE_PATH']=resources
  with (folder/'server.log').open('w') as log:proc=subprocess.Popen(['ign','gazebo','-s','-r',str(world)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  self.state.write_text(json.dumps(dict(pid=proc.pid,world=str(world),variant=variant),indent=2)+'\n')
  # Service readiness is checked, never inferred from a fixed sleep.
  deadline=time.monotonic()+90
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise RuntimeError('Gazebo exited; see '+str(folder/'server.log'))
   check=subprocess.run(['ign','service','-s','/world/pig_pen_16units_world/control','--reqtype','ignition.msgs.WorldControl','--reptype','ignition.msgs.Boolean','--timeout','1000','--req','pause: true'],capture_output=True,text=True,env=env,timeout=4)
   if 'data: true' in check.stdout:
    variant.update(world=str(world),world_sha256=digest(world));return variant
   time.sleep(.5)
  raise RuntimeError('Experiment scene did not become ready')
