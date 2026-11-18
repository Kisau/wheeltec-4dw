#!/usr/bin/env python3
"""Generate a raised pig-pen world with physical roughness and entrance ramps."""
import argparse
import json
import math
import random
import struct
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAISE = .06
RAMP_RUN = .15
MAIN_HALF = .60
MAIN_END = 11.865
BRANCHES = ((-6.675, -5.685), (-.495, .495), (5.685, 6.675))


def png(path, rows, size=256):
    def chunk(tag,data):
        return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',size,size,8,2,0,0,0))
                     +chunk(b'IDAT',zlib.compress(b''.join(b'\0'+bytes(row) for row in rows)))
                     +chunk(b'IEND',b''))


def gravel_textures(directory):
    """Packed, randomly placed angular pebbles over fines; no cell grid."""
    size, tile = 512, .128
    rng = random.Random(47)
    stones, buckets = [], {}
    bucket_size = .01
    bucket_count = math.ceil(tile / bucket_size)
    # Periodic minimum-spacing placement. Reject overlaps, never snap to rows.
    for _ in range(22000):
        x,y = rng.random()*tile, rng.random()*tile
        radius = rng.uniform(.0025,.005)
        nearby = []
        # Grid is only a search accelerator; positions remain continuous.
        for dx in (-2,-1,0,1,2):
            for dy in (-2,-1,0,1,2):
                key=((int(x/tile*bucket_count)+dx)%bucket_count,
                     (int(y/tile*bucket_count)+dy)%bucket_count)
                nearby.extend(buckets.get(key,()))
        if any(math.hypot(min(abs(x-a),tile-abs(x-a)),
                          min(abs(y-b),tile-abs(y-b))) < .92*(radius+r)
               for a,b,r in nearby):
            continue
        stone=(x,y,radius)
        stones.append(stone)
        key=(int(x/tile*bucket_count),int(y/tile*bucket_count))
        buckets.setdefault(key,[]).append(stone)
        if len(stones)>=320:
            break
    heights=[[0. for _ in range(size)] for _ in range(size)]
    colours=[]
    for y in range(size):
        row=[]
        for x in range(size):
            base=rng.randrange(69,91)
            row.extend((base+7,base+4,base))
        colours.append(row)
    records=[]
    for x,y,radius in stones:
        sides=rng.randrange(6,10);rotation=rng.random()*math.tau
        polygon=[]
        for k in range(sides):
            angle=rotation+math.tau*k/sides
            r=radius*rng.uniform(.78,1.)
            polygon.append((math.cos(angle)*r,math.sin(angle)*r))
        peak=rng.uniform(.001,.003)
        base=rng.randrange(100,181)
        tint=rng.choice(((0,1,3),(8,5,0),(1,1,0),(-3,0,5)))
        slope_x,slope_y=rng.uniform(-.4,.4),rng.uniform(-.4,.4)
        records.append(dict(centre_m=[x,y],diameter_m=2*radius,exposed_height_m=peak))
        for oy in (-tile,0,tile):
            for ox in (-tile,0,tile):
                cx,cy=x+ox,y+oy
                x0=max(0,int((cx-radius)/tile*size));x1=min(size-1,int((cx+radius)/tile*size)+1)
                y0=max(0,int((cy-radius)/tile*size));y1=min(size-1,int((cy+radius)/tile*size)+1)
                for py in range(y0,y1+1):
                    for px in range(x0,x1+1):
                        u,v=(px+.5)*tile/size-cx,(py+.5)*tile/size-cy
                        inside=False
                        for (ax,ay),(bx,by) in zip(polygon,polygon[1:]+polygon[:1]):
                            if (ay>v)!=(by>v) and u<(bx-ax)*(v-ay)/(by-ay)+ax:
                                inside=not inside
                        if not inside:
                            continue
                        edge=min(1.,max(0.,(1-math.hypot(u,v)/radius)*5))
                        facet=max(.35,min(1.,.78+slope_x*u/radius+slope_y*v/radius))
                        h=peak*edge*facet
                        if h<heights[py][px]:
                            continue
                        heights[py][px]=h
                        speckle=rng.randrange(-9,10)
                        shade=base*(.55+.45*facet)*(.72+.28*edge)+speckle
                        for channel in range(3):
                            colours[py][px*3+channel]=int(max(0,min(255,shade+tint[channel])))
    normals=[]
    spacing=2*tile/size
    for y in range(size):
        row=[]
        for x in range(size):
            dx=(heights[y][(x+1)%size]-heights[y][(x-1)%size])/spacing
            dy=(heights[(y+1)%size][x]-heights[(y-1)%size][x])/spacing
            length=math.sqrt(dx*dx+dy*dy+1)
            row.extend(int(127.5*(n/length+1)) for n in (-dx,-dy,1.))
        normals.append(row)
    png(directory/'gravel_irregular_albedo.png',colours,size)
    png(directory/'gravel_irregular_normal.png',normals,size)
    (directory/'gravel_material.json').write_text(json.dumps(dict(
        tile_m=tile,placement='periodic random packing, irregular polygons over fines',
        representation='visual and normal detail; aggregate collision unchanged',
        stones=records),indent=2)+'\n')


def write_mesh(path, x0, x1, y0, y1, height, step=.08):
    """Closed triangle mesh; neighbouring patches share exact top heights."""
    nx, ny = math.ceil((x1 - x0) / step), math.ceil((y1 - y0) / step)
    vertices = [(x0 + (x1-x0)*i/nx, y0 + (y1-y0)*j/ny,
                 height(x0 + (x1-x0)*i/nx, y0 + (y1-y0)*j/ny))
                for j in range(ny+1) for i in range(nx+1)]
    count = len(vertices)
    vertices += [(x, y, -.03) for x, y, _ in vertices]
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            b, c, d = a+1, a+nx+1, a+nx+2
            faces += [(a,b,d), (a,d,c), (a+count,d+count,b+count),
                      (a+count,c+count,d+count)]
    boundary = (list(range(nx+1))
                + [j*(nx+1)+nx for j in range(1,ny+1)]
                + [ny*(nx+1)+i for i in range(nx-1,-1,-1)]
                + [j*(nx+1) for j in range(ny-1,0,-1)])
    for a,b in zip(boundary, boundary[1:]+boundary[:1]):
        faces += [(a,a+count,b+count), (a,b+count,b)]
    normals = [[0.,0.,0.] for _ in vertices]
    for face in faces:
        a,b,c=[vertices[i] for i in face]
        u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
        normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        for index in face:
            for axis in range(3): normals[index][axis] += normal[axis]
    with path.open('w') as f:
        for v in vertices:
            f.write('v ' + ' '.join(f'{n:.8f}' for n in v) + '\n')
        for x,y,z in vertices:
            f.write(f'vt {x/.128:.8f} {y/.128:.8f}\n')
        for normal in normals:
            length=math.sqrt(sum(n*n for n in normal))
            f.write('vn '+' '.join(f'{n/length:.8f}' for n in normal)+'\n')
        for face in faces:
            f.write('f ' + ' '.join(f'{i+1}/{i+1}/{i+1}' for i in face) + '\n')
    # Keep UV-rich OBJ for rendering, and a plain triangle STL for the older
    # DART/ODE collision importer (which can crash on UV-only OBJ meshes).
    with path.with_suffix('.stl').open('wb') as f:
        f.write(b'rough_pen_collision'.ljust(80,b'\0'))
        f.write(struct.pack('<I',len(faces)))
        for face in faces:
            a,b,c=[vertices[i] for i in face]
            u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
            normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
            length=math.sqrt(sum(x*x for x in normal))
            normal=[x/length for x in normal]
            f.write(struct.pack('<12fH',*normal,*a,*b,*c,0))
    return len(vertices), len(faces)


def physical_gravel(path, bounds, base_height, material, grains_per_tile=96, amplitude=.003):
    """Merge CLOSED, embedded angular grains into the actual collision STL.

    A grain is a hexagonal double pyramid: the lower tip is embedded in the
    base and its upper tip protrudes 1--3 mm. No artificial body forces.
    """
    import shutil
    target=path.with_name(path.stem+'_gravel.stl')
    shutil.copyfile(path.with_suffix('.stl'),target)
    data=json.loads(material.read_text())
    tile=data['tile_m']; grains=data['stones'][:grains_per_tile]
    x0,x1,y0,y1=bounds
    base_triangles=struct.unpack('<I',target.read_bytes()[80:84])[0]
    count=0; first=None
    with target.open('ab') as output:
        for j in range(math.floor(y0/tile),math.ceil(y1/tile)):
            for i in range(math.floor(x0/tile),math.ceil(x1/tile)):
                for index,grain in enumerate(grains):
                    gx,gy=grain['centre_m'];x,y=i*tile+gx,j*tile+gy
                    radius=grain['diameter_m']/2
                    if not (x0+radius<=x<=x1-radius and y0+radius<=y<=y1-radius):continue
                    peak=grain['exposed_height_m']*amplitude/.003
                    # Stable, non-circular outlines, no aligned rows of stones.
                    rng=random.Random((i*73856093 ^ j*19349663 ^ index*83492791)&0xffffffff)
                    angle=rng.random()*math.tau
                    ring=[(x+radius*math.cos(angle+k*math.tau/6),
                           y+radius*math.sin(angle+k*math.tau/6),base_height) for k in range(6)]
                    top=(x,y,base_height+peak);bottom=(x,y,base_height-.001)
                    for k in range(6):
                        for a,b,c in ((ring[k],ring[(k+1)%6],top),
                                      (ring[(k+1)%6],ring[k],bottom)):
                            u=[b[q]-a[q] for q in range(3)];v=[c[q]-a[q] for q in range(3)]
                            normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                            length=math.sqrt(sum(n*n for n in normal))
                            output.write(struct.pack('<12fH',*(n/length for n in normal),*a,*b,*c,0))
                    if first is None:first=dict(centre_m=[x,y],base_height_m=base_height,
                                                diameter_m=2*radius,exposed_height_m=peak)
                    count+=1
    triangles=base_triangles+12*count
    with target.open('r+b') as output:
        output.seek(80);output.write(struct.pack('<I',triangles))
    return dict(grains=count,triangles=triangles,base_triangles=base_triangles,
                collision_file=target.name,probe_grain=first)


def mesh_model(name, mesh, friction, colour, gravel=False):
    m = ET.Element('model', name=name)
    ET.SubElement(m, 'static').text = 'true'
    link = ET.SubElement(m, 'link', name='terrain')
    for kind in ('collision', 'visual'):
        part = ET.SubElement(link, kind, name=kind)
        shape = ET.SubElement(ET.SubElement(part, 'geometry'), 'mesh')
        filename=(mesh.stem+'_gravel.stl' if gravel else
                  mesh.with_suffix('.stl').name if kind == 'collision' else mesh.name)
        ET.SubElement(shape, 'uri').text = 'meshes/rough_pen/' + filename
        if kind == 'collision':
            ode = ET.SubElement(ET.SubElement(ET.SubElement(part, 'surface'), 'friction'), 'ode')
            ET.SubElement(ode, 'mu').text = str(friction)
            ET.SubElement(ode, 'mu2').text = str(friction)
        else:
            mat = ET.SubElement(part, 'material')
            for tag in ('ambient', 'diffuse'):
                ET.SubElement(mat, tag).text = colour
            if name.startswith('rough_') and not gravel:
                metal=ET.SubElement(ET.SubElement(mat,'pbr'),'metal')
                ET.SubElement(metal,'albedo_map').text='meshes/rough_pen/gravel_irregular_albedo.png'
                ET.SubElement(metal,'normal_map').text='meshes/rough_pen/gravel_irregular_normal.png'
                ET.SubElement(metal,'roughness').text='.95'
                ET.SubElement(metal,'metalness').text='0'
    return m


def build(source, output, amplitude=.003, grains_per_tile=96):
    if not math.isfinite(amplitude) or not 0 <= amplitude <= .01:
        raise ValueError('roughness amplitude must be in [0, 0.01] metres')
    if type(grains_per_tile) is not int or not 0<=grains_per_tile<=295:
        raise ValueError('grains_per_tile must be an integer between 0 and 295')
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    if world.find("model[@name='rough_main_0_west']") is not None:
        raise ValueError('Source must be the baseline world, not generated rough world')
    directory = output.parent / 'meshes' / 'rough_pen'
    directory.mkdir(parents=True, exist_ok=True)
    gravel_textures(directory)
    # Lower fallback plane so it cannot flatten branch depressions.
    ET.SubElement(world.find("model[@name='ground_plane']"), 'pose').text = '0 0 -0.03 0 0 0'
    for include in world.findall('include'):
        pose=include.find('pose')
        if pose is not None:
            values=list(map(float,pose.text.split()));values[2]+=RAISE
            pose.text=' '.join(map(str,values))
    for model in world.findall('model'):
        name = model.get('name', '')
        if (name.startswith('pen') and name[3:].isdigit()) or name.startswith('junction_iron_post_'):
            pose = model.find('pose')
            values = list(map(float, pose.text.split()))
            values[2] += RAISE
            pose.text = ' '.join(f'{v:.6f}' for v in values)
            if name.startswith('pen'):
                foundation = ET.SubElement(model, 'link', name='raised_foundation')
                ET.SubElement(foundation, 'pose').text = '0 0 -0.03 0 0 0'
                for kind in ('collision','visual'):
                    part = ET.SubElement(foundation, kind, name=kind)
                    ET.SubElement(ET.SubElement(ET.SubElement(part,'geometry'),'box'),'size').text = '3.30 2.55 0.06'
        elif name == 'corridor_end_walls':
            for link in model.findall('link'):
                if link.get('name','').startswith('main_'):
                    pose = link.find('pose'); values = list(map(float,pose.text.split()))
                    values[2] += RAISE
                    pose.text = ' '.join(map(str,values))
    records = []
    def add(name, bounds, height, friction, colour):
        mesh = directory / (name+'.obj')
        vertices, triangles = write_mesh(mesh, *bounds, height)
        gravel=name.startswith('rough_')
        physical=(physical_gravel(mesh,bounds,RAISE if name.startswith('rough_main') else 0,
                                 directory/'gravel_material.json',grains_per_tile,amplitude)
                  if gravel else None)
        world.append(mesh_model(name, mesh, friction, colour,gravel=gravel))
        records.append(dict(name=name, bounds=bounds, friction=friction,
                            vertices=vertices, triangles=triangles,physical_gravel=physical))
    def main_height(x,y):
        return RAISE
    # Six friction patches retain a continuous top surface, no artificial steps.
    cuts = (-MAIN_END,-3.09,3.09,MAIN_END)
    for j in range(3):
        for side,(x0,x1),mu in [('west',(-.6,0.),.65),('east',(0.,.6),.85)]:
            add(f'rough_main_{j}_{side}',(x0,x1,cuts[j],cuts[j+1]),main_height,
                mu if j != 1 else mu-.1,'0.43 0.40 0.35 1')
    for j,(lo,hi) in enumerate(BRANCHES,1):
        for side,sign in [('west',-1),('east',1)]:
            x0,x1=sorted((sign*.6,sign*.75))
            add(f'entrance_ramp_{j}_{side}',(x0,x1,lo,hi),
                lambda x,y: RAISE*(.75-abs(x))/RAMP_RUN,.75,'0.65 0.56 0.36 1')
            x0,x1=sorted((sign*.75,sign*3.825))
            def branch_height(x,y,lo=lo,hi=hi):
                return 0.
            add(f'rough_branch_{j}_{side}',(x0,x1,lo,hi),branch_height,
                .60 if sign < 0 else .80,'0.47 0.43 0.37 1')
    ET.indent(tree,space='  ')
    tree.write(output,encoding='utf-8',xml_declaration=True)
    manifest=dict(source=source.name, world=output.name, raised_m=RAISE,
                  ramp_run_m=RAMP_RUN, ramp_slope_deg=math.degrees(math.atan(RAISE/RAMP_RUN)),
                  roughness_amplitude_m=amplitude, friction_range=[.55,.85],
                  gravel_diameter_m=[.005,.010], gravel_exposed_height_m=[.001,.003],
                  gravel_texture_tile_m=.128,
                  contact_model='explicit closed 3D grains merged into static collision and visual STL; embedded, not loose bodies',
                  grains_per_tile=grains_per_tile,
                  physical_grains=sum((r['physical_gravel'] or {}).get('grains',0) for r in records),
                  spawn_z=.11, calibration='provisional, not measured real-floor parameters',
                  surfaces=records)
    output.with_suffix('.terrain.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'worlds/pig_pen_16units.world')
    p.add_argument('--output',type=Path,default=ROOT/'worlds/pig_pen_16units_rough.world')
    p.add_argument('--roughness',type=float,default=.003)
    p.add_argument('--grains-per-tile',type=int,default=96)
    a=p.parse_args(); result=build(a.source,a.output,a.roughness,a.grains_per_tile)
    print(f"Generated {a.output}: {len(result['surfaces'])} surfaces, six 6 cm / 15 cm ramps")
