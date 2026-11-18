#!/usr/bin/env python3
"""Build a flat, constant-width U corridor. Dimensions follow the centreline."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def build():
    sdf = ET.Element('sdf', version='1.9')
    world = ET.SubElement(sdf, 'world', name='u_corridor_world')
    ET.SubElement(world, 'gravity').text = '0 0 -9.8'
    physics = ET.SubElement(world, 'physics', name='default', type='ignored')
    ET.SubElement(physics, 'max_step_size').text = '0.001'
    ET.SubElement(physics, 'real_time_factor').text = '1'
    for system in ('Physics', 'Sensors', 'UserCommands', 'SceneBroadcaster'):
        filenames = {'Physics': 'physics', 'Sensors': 'sensors',
                     'UserCommands': 'user-commands', 'SceneBroadcaster': 'scene-broadcaster'}
        plugin = ET.SubElement(world, 'plugin', filename=f'gz-sim-{filenames[system]}-system',
                               name=f'gz::sim::systems::{system}')
        if system == 'Sensors':
            ET.SubElement(plugin, 'render_engine').text = 'ogre2'
    light = ET.SubElement(world, 'light', name='sun', type='directional')
    for key, value in {'pose': '0 0 15 0 0 0', 'diffuse': '.9 .9 .9 1',
                       'specular': '.2 .2 .2 1', 'direction': '-.3 .2 -1'}.items():
        ET.SubElement(light, key).text = value

    def box(name, x, y, z, sx, sy, sz, color):
        model = ET.SubElement(world, 'model', name=name)
        ET.SubElement(model, 'static').text = 'true'
        ET.SubElement(model, 'pose').text = f'{x:g} {y:g} {z:g} 0 0 0'
        link = ET.SubElement(model, 'link', name='link')
        for kind in ('collision', 'visual'):
            item = ET.SubElement(link, kind, name=kind)
            shape = ET.SubElement(ET.SubElement(item, 'geometry'), 'box')
            ET.SubElement(shape, 'size').text = f'{sx:g} {sy:g} {sz:g}'
            if kind == 'visual':
                material = ET.SubElement(item, 'material')
                for prop in ('ambient', 'diffuse'):
                    ET.SubElement(material, prop).text = color
            else:
                ode = ET.SubElement(ET.SubElement(ET.SubElement(item, 'surface'), 'friction'), 'ode')
                ET.SubElement(ode, 'mu').text = '1'
                ET.SubElement(ode, 'mu2').text = '1'

    box('flat_ground', 11, .2, -.1, 32, 34, .2, '.65 .65 .65 1')
    # Road faces: x=-1.2,1.2,20.8,23.2; top y=10.5,12.9;
    # bottom end faces y=-12.5. All walls are .1m thick and 1m tall.
    walls = [
        ('outer_left', -1.25, .2, .1, 25.4),
        ('outer_right', 23.25, .2, .1, 25.4),
        ('outer_top', 11, 12.95, 24.6, .1),
        ('inner_left', 1.25, -1, .1, 23),
        ('inner_right', 20.75, -1, .1, 23),
        ('inner_top', 11, 10.45, 19.6, .1),
        ('end_A', 0, -12.55, 2.4, .1),
        ('end_D', 22, -12.55, 2.4, .1),
    ]
    for name, x, y, sx, sy in walls:
        box(name, x, y, .5, sx, sy, 1, '.28 .4 .55 1')
    ET.indent(sdf, space='  ')
    ET.ElementTree(sdf).write(ROOT / 'worlds/u_corridor_flat.world',
                              encoding='utf-8', xml_declaration=True)
    route = {
        'world': 'u_corridor_world', 'units': 'metres',
        'length_reference': 'centreline', 'clear_width': 2.4,
        'side_length': 23.0, 'top_length': 22.0,
        'wall_height': 1.0, 'wall_thickness': .1,
        'spawn': {'x': 0, 'y': -11.3, 'z': .05, 'yaw': 1.5707963267948966},
        'waypoints': {'A': [0, -11.3], 'B': [0, 11.7],
                      'C': [22, 11.7], 'D': [22, -11.3]},
        'outbound': ['A', 'B', 'C', 'D'], 'outbound_direction': 'forward',
        'inbound': ['D', 'C', 'B', 'A'], 'inbound_direction': 'reverse',
        'one_way_length': 68.0, 'round_trip_length': 136.0,
        'endpoint_wall_clearance': 1.2,
    }
    (ROOT / 'maps/u_corridor/route.json').write_text(json.dumps(route, indent=2) + '\n')


if __name__ == '__main__':
    build()
