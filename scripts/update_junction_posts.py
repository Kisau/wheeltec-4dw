#!/usr/bin/env python3
"""Replace existing junction posts with mirrored, joined U-channel sections."""
import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

THICKNESS = .002  # Provisional steel thickness; outer envelope stays 4 x 8 cm.


def channel_post(name, pose):
    north = name.endswith('_north')
    east = '_east_' in name
    toward_junction = -1 if north else 1
    toward_main = -1 if east else 1
    near_y = toward_junction * .02
    far_y = -near_y
    # Each tuple is (name, centre X, centre Y, size X, size Y).
    plates = [
        ('near_left', -.02 + THICKNESS / 2, near_y, THICKNESS, .04),
        ('near_right', .02 - THICKNESS / 2, near_y, THICKNESS, .04),
        ('near_back', 0., near_y - toward_junction * (.02 - THICKNESS / 2),
         .04, THICKNESS),
        ('far_south', 0., far_y - .02 + THICKNESS / 2, .04, THICKNESS),
        ('far_north', 0., far_y + .02 - THICKNESS / 2, .04, THICKNESS),
        ('far_back', -toward_main * (.02 - THICKNESS / 2), far_y,
         THICKNESS, .04),
    ]
    model = ET.Element('model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = pose
    link = ET.SubElement(model, 'link', name='post')
    for label, x, y, sx, sy in plates:
        for kind in ('collision', 'visual'):
            part = ET.SubElement(link, kind, name=label)
            ET.SubElement(part, 'pose').text = f'{x:.4f} {y:.4f} 0 0 0 0'
            box = ET.SubElement(ET.SubElement(part, 'geometry'), 'box')
            ET.SubElement(box, 'size').text = f'{sx:.4f} {sy:.4f} 0.8300'
            if kind == 'visual':
                material = ET.SubElement(part, 'material')
                ET.SubElement(material, 'ambient').text = '0.25 0.27 0.30 1'
                ET.SubElement(material, 'diffuse').text = '0.35 0.37 0.40 1'
                ET.SubElement(material, 'specular').text = '0.65 0.65 0.65 1'
    return model


def update(path):
    source = path.read_text()
    pattern = r'    <model name="(junction_iron_post_[^"]+)">.*?</model>'
    def replace(match):
        old = ET.fromstring(match.group(0))
        model = channel_post(match.group(1), old.findtext('pose'))
        ET.indent(model, space='  ', level=2)
        return '    ' + ET.tostring(model, encoding='unicode').rstrip()
    result, count = re.subn(pattern, replace, source, flags=re.S)
    if count != 12:
        raise ValueError(f'Expected 12 junction posts, got {count}')
    ET.fromstring(result)
    path.write_text(result)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    args = parser.parse_args()
    print(f'Updated {update(args.world)} double U-channel posts (2 mm steel)')
