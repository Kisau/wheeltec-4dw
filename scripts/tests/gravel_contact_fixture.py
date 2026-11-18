#!/usr/bin/env python3
"""Create a constrained-sphere physical gravel / flat-ground contact fixture.

Run the generated probe.world in an isolated Gazebo partition, then compare
model Z positions with expected.json. Both balls have radius 2 mm; a vertical
prismatic joint prevents the gravel ball from rolling off its grain.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_rough_pen_world as b


def make_fixture(root):
    root.mkdir(parents=True,exist_ok=True);material=b.ROOT/'worlds/meshes/rough_pen/gravel_material.json'
    mesh=root/'probe_base.obj';b.write_mesh(mesh,-.064,.064,-.064,.064,lambda x,y:.06)
    g=b.physical_gravel(mesh,(-.064,.064,-.064,.064),.06,material,32)
    x,y=g['probe_grain']['centre_m'];height=g['probe_grain']['exposed_height_m']
    model=''
    for name,px,py in [('grain_probe',x,y),('flat_probe',.2,0)]:
     model+=f'''<model name="{name}"><pose>{px} {py} .075 0 0 0</pose>
    <link name="ball"><inertial><mass>0.02</mass><inertia><ixx>0.000000032</ixx><iyy>0.000000032</iyy><izz>0.000000032</izz></inertia></inertial>
    <collision name="ball"><geometry><sphere><radius>.002</radius></sphere></geometry></collision></link>
    <joint name="vertical" type="prismatic"><parent>world</parent><child>ball</child><axis><xyz>0 0 1</xyz><limit><lower>-1</lower><upper>1</upper></limit></axis></joint></model>'''
    world=f'''<sdf version="1.9"><world name="gravel_contact_probe"><gravity>0 0 -9.8</gravity>
    <physics name="p" type="ignored"><max_step_size>.001</max_step_size><real_time_factor>1</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <model name="gravel"><static>true</static><link name="surface"><collision name="terrain"><geometry><mesh><uri>{mesh.with_name('probe_base_gravel.stl').as_uri()}</uri></mesh></geometry></collision></link></model>
    <model name="flat_reference"><static>true</static><pose>.2 0 .03 0 0 0</pose><link name="surface"><collision name="flat"><geometry><box><size>.04 .04 .06</size></box></geometry></collision></link></model>
    {model}</world></sdf>'''
    (root/'probe.world').write_text(world);(root/'expected.json').write_text(json.dumps(dict(grain_height_m=height,expected_flat_centre_z=.062,expected_grain_centre_z=.062+height)))
    return g['probe_grain']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/tmp/gravel_contact_probe'))
    args=parser.parse_args()
    print(make_fixture(args.output))
