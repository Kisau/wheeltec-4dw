import collections
import importlib.util
import math
import json
import struct
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / 'build_rough_pen_world.py'
spec = importlib.util.spec_from_file_location('rough_world', SCRIPT)
rough = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rough)


class RoughWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / 'rough.world'
        cls.manifest = rough.build(rough.ROOT/'worlds/pig_pen_16units.world', cls.output, grains_per_tile=4)
        cls.world = ET.parse(cls.output).getroot().find('world')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def mesh(self, name):
        vertices, faces = [], []
        for row in (self.output.parent/'meshes/rough_pen'/f'{name}.obj').read_text().splitlines():
            fields = row.split()
            if fields[0] == 'v': vertices.append(tuple(map(float, fields[1:])))
            elif fields[0] == 'f': faces.append(tuple(int(x.split('/')[0])-1 for x in fields[1:]))
        return vertices, faces

    def test_raised_pens_posts_and_main_end_walls(self):
        baseline = ET.parse(rough.ROOT/'worlds/pig_pen_16units.world').getroot().find('world')
        for old in baseline.findall('model'):
            name = old.get('name')
            if (name.startswith('pen') and name[3:].isdigit()) or name.startswith('junction_iron_post_'):
                new = self.world.find(f"model[@name='{name}']")
                self.assertAlmostEqual(float(new.findtext('pose').split()[2])-
                                       float(old.findtext('pose').split()[2]), .06)
        self.assertAlmostEqual(self.manifest['ramp_slope_deg'], 21.801409486)

    def test_ramps_join_main_and_branch_without_steps(self):
        ramps = [r for r in self.manifest['surfaces'] if r['name'].startswith('entrance_ramp')]
        self.assertEqual(len(ramps), 6)
        for ramp in ramps:
            vertices, _ = self.mesh(ramp['name'])
            top = [v for v in vertices if v[2] > -.029]
            self.assertAlmostEqual(ramp['bounds'][1]-ramp['bounds'][0], .15)
            self.assertAlmostEqual(ramp['bounds'][3]-ramp['bounds'][2], .99)
            for x, y, z in top:
                self.assertAlmostEqual(z, .06*(.75-abs(x))/.15, places=7)
            self.assertTrue(any(abs(abs(x)-.6)<1e-8 and abs(z-.06)<1e-8 for x,y,z in top))
            self.assertTrue(any(abs(abs(x)-.75)<1e-8 and abs(z)<1e-8 for x,y,z in top))

    def test_closed_meshes_and_identical_visual_collision(self):
        for surface in self.manifest['surfaces']:
            vertices, faces = self.mesh(surface['name'])
            edges = collections.Counter()
            for face in faces:
                for a,b in zip(face, face[1:]+face[:1]):
                    edges[tuple(sorted((a,b)))] += 1
            self.assertTrue(all(count == 2 for count in edges.values()), surface['name'])
            model = self.world.find(f"model[@name='{surface['name']}']")
            collision = Path(model.findtext('link/collision/geometry/mesh/uri'))
            visual = Path(model.findtext('link/visual/geometry/mesh/uri'))
            self.assertEqual(collision.stem, visual.stem)
            stl=(self.output.parent/collision).read_bytes()
            physical=surface['physical_gravel']
            expected=physical['triangles'] if physical else len(faces)
            self.assertEqual(struct.unpack('<I',stl[80:84])[0],expected)
            self.assertEqual(len(stl),84+50*expected)
            self.assertTrue(.55 <= surface['friction'] <= .85)
            nominal = .06 if surface['name'].startswith('rough_main') else 0
            if not surface['name'].startswith('entrance_ramp'):
                top = vertices[:len(vertices)//2]
                self.assertTrue(all(abs(z-nominal)<=.00300001 for x,y,z in top))
                self.assertEqual(max(z for x,y,z in top),min(z for x,y,z in top))

    def test_actual_collision_contains_closed_millimetre_grains_above_flat_base(self):
        self.assertGreater(self.manifest['physical_grains'],100)
        for surface in self.manifest['surfaces']:
            physical=surface['physical_gravel']
            if not physical:continue
            data=(self.output.parent/'meshes/rough_pen'/physical['collision_file']).read_bytes()
            offset=84+50*physical['base_triangles']
            triangles=[]
            for k in range(12):
                record=struct.unpack_from('<12fH',data,offset+50*k)
                triangles.append(tuple(tuple(round(record[3+3*j+i],7) for i in range(3)) for j in range(3)))
            edges=collections.Counter()
            for triangle in triangles:
                for a,b in zip(triangle,triangle[1:]+triangle[:1]):edges[tuple(sorted((a,b)))]+=1
            self.assertTrue(all(n==2 for n in edges.values()))
            vertices={v for t in triangles for v in t}
            base=physical['probe_grain']['base_height_m']
            exposed=max(v[2] for v in vertices)-base
            diameter=max(math.hypot(a[0]-b[0],a[1]-b[1]) for a in vertices for b in vertices)
            self.assertTrue(.0009999<=exposed<=.0030001)
            self.assertTrue(.0049999<=diameter<=.0100001)
            self.assertTrue(any(v[2]<base for v in vertices))

    def test_fine_gravel_has_scaled_materials_and_physical_collision(self):
        self.assertEqual(self.manifest['gravel_diameter_m'], [.005,.010])
        self.assertEqual(self.manifest['gravel_exposed_height_m'], [.001,.003])
        self.assertEqual(self.manifest['roughness_amplitude_m'], .003)
        for name in ('gravel_irregular_albedo.png','gravel_irregular_normal.png'):
            self.assertEqual((self.output.parent/'meshes/rough_pen'/name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        material=json.loads((self.output.parent/'meshes/rough_pen/gravel_material.json').read_text())
        self.assertGreater(len(material['stones']),200)
        for stone in material['stones']:
            self.assertTrue(.005 <= stone['diameter_m'] <= .010)
            self.assertTrue(.001 <= stone['exposed_height_m'] <= .003)
        self.assertEqual(len({round(s['centre_m'][0],7) for s in material['stones']}),len(material['stones']))
        self.assertEqual(len({round(s['centre_m'][1],7) for s in material['stones']}),len(material['stones']))

    def test_prepared_scene_retains_mesh_and_texture_resources(self):
        prepare_spec = importlib.util.spec_from_file_location('prepare_scene', SCRIPT.with_name('prepare_training_scene.py'))
        prepare = importlib.util.module_from_spec(prepare_spec)
        prepare_spec.loader.exec_module(prepare)
        target = self.output.parent/'prepared'
        prepare.prepare(self.output, rough.ROOT/'turn_on_wheeltec_robot/urdf/four_wheel_diff_bs_robot.urdf',
                        target, rtf=2, lidar_only=True)
        prepared = ET.parse(target/'training.world')
        for element in prepared.getroot().iter():
            if element.tag in ('uri','albedo_map','normal_map') and 'meshes/rough_pen/' in (element.text or ''):
                self.assertTrue(element.text.startswith('file:///'))
                self.assertTrue(Path(element.text.removeprefix('file://')).is_file())


if __name__ == '__main__':
    unittest.main()
