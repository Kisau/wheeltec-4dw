"""Generate actual wheel-friction/base-payload variants, and respawn in Gazebo."""
from pathlib import Path
import math,subprocess,xml.etree.ElementTree as ET
from phase_junction_bc import digest

def robot_variant(source,target,friction=1.,payload_kg=0.):
 if not .2<=friction<=1.5 or not 0<=payload_kg<=10:raise ValueError('Unsupported dynamics range')
 tree=ET.parse(source);root=tree.getroot();inertial=root.find("link[@name='base_link']/inertial")
 if inertial is None:raise ValueError('Base inertial missing')
 mass=inertial.find('mass');old=float(mass.get('value'));mass.set('value',str(old+payload_kg))
 # Approximate extra uniformly distributed payload at the existing base COM.
 for key,value in inertial.find('inertia').attrib.items():inertial.find('inertia').set(key,str(float(value)*(old+payload_kg)/old))
 n=0
 for gazebo in root.findall('gazebo'):
  if gazebo.find('mu1') is None or gazebo.find('mu2') is None:continue
  for tag in ('mu1','mu2'):
   node=gazebo.find(tag)
   if node is not None:node.text=str(friction);n+=1
 if n!=8:raise ValueError(f'Expected eight wheel friction coefficients, found {n}')
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);tree.write(target,encoding='utf-8',xml_declaration=True)
 return dict(friction=friction,payload_kg=payload_kg,base_mass_kg=old+payload_kg,path=str(target.resolve()),sha256=digest(target))

def service(name,request_type,request):
 cmd=['ign','service','-s','/world/pig_pen_16units_world/'+name,'--reqtype',request_type,'--reptype','ignition.msgs.Boolean','--timeout','10000','--req',request]
 out=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 if out.returncode or 'data: true' not in out.stdout:raise RuntimeError('Gazebo '+name+' failed: '+out.stdout+out.stderr)

def respawn(env,robot):
 env._publish((0.,0.));env._drain();env._control()
 service('remove','ignition.msgs.Entity','name: "wheeltec_mini" type: MODEL');env._control(steps=1)
 # Protobuf string escaping, not shell interpolation.
 import json
 request='sdf_filename: '+json.dumps(str(Path(robot).resolve()))+' name: "wheeltec_mini" allow_renaming: false pose { position { x: 0 y: -11.3 z: 0.17 } orientation { z: '+str(math.sin(math.pi/4))+' w: '+str(math.cos(math.pi/4))+' } }'
 service('create','ignition.msgs.EntityFactory',request);env._control(steps=1);env._drain()
 for _ in range(5):env._advance(.2)

def verify_world_variant(expected,output):
 """Read live generated world SDF; do not trust only the source URDF."""
 import ast
 cmd=['ign','service','-s','/world/pig_pen_16units_world/state','--reqtype','ignition.msgs.Empty','--reptype','ignition.msgs.SerializedStepMap','--timeout','10000','--req',' ']
 out=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 if out.returncode:raise RuntimeError('Cannot read live entity state: '+out.stderr)
 # Live entity ModelSdf component: extracted from Gazebo's state response,
 # not the file that was submitted for spawning.
 sdf=None
 for line in out.stdout.splitlines():
  if 'component:' in line and '<?xml' in line and 'wheeltec_mini' in line:
   candidate=ast.literal_eval(line.split('component:',1)[1].strip())
   if ET.fromstring(candidate).find("model[@name='wheeltec_mini']") is not None:
    sdf=candidate;break
 if sdf is None:raise RuntimeError('Live robot ModelSdf component missing')
 Path(output).write_text(sdf)
 tree=ET.fromstring(sdf);model=tree.find(".//model[@name='wheeltec_mini']")
 if model is None:raise RuntimeError('Spawned model missing from live SDF')
 masses=[float(x.text) for x in model.findall('.//inertial/mass')]
 source=ET.parse(expected['path'])
 wheels=[g.get('reference') for g in source.findall('gazebo') if g.find('mu1') is not None]
 mus=[float(x.text) for link in model.findall('link') if link.get('name') in wheels for x in link.findall('.//surface/friction/ode/mu')]
 mus2=[float(x.text) for link in model.findall('link') if link.get('name') in wheels for x in link.findall('.//surface/friction/ode/mu2')]
 # Fixed joints may be lumped into the base; compare total physical mass.
 source=ET.parse(expected['path']);wanted=sum(float(x.get('value')) for x in source.findall('.//inertial/mass'))
 if abs(sum(masses)-wanted)>.05 or len(mus)<4 or any(abs(x-expected['friction'])>.001 for x in mus+mus2):
  raise RuntimeError(f'Live dynamics mismatch: masses={masses}, mus={mus}, expected total={wanted}')
 return dict(total_mass_kg=sum(masses),wheel_mu=mus,wheel_mu2=mus2,live_sdf=str(Path(output).resolve()),sha256=digest(output))
