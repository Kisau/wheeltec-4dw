"""Small-angle point-to-line map matching; deterministic geometry, no learning."""
import math
import xml.etree.ElementTree as ET
import numpy as np

from u_corridor_fsm import wrap


class WallLocalizer:
    def __init__(self, world_path, spawn):
        self.segments=[]
        ox,oy,yaw=spawn
        c,s=math.cos(yaw),math.sin(yaw)
        def local(x,y):return (c*(x-ox)+s*(y-oy),-s*(x-ox)+c*(y-oy))
        for m in ET.parse(world_path).findall('.//world/model'):
            if m.get('name')=='flat_ground':continue
            x,y,*_=map(float,m.findtext('pose').split())
            sx,sy,_=map(float,m.findtext('link/collision/geometry/box/size').split())
            corners=[local(x-sx/2,y-sy/2),local(x+sx/2,y-sy/2),
                     local(x+sx/2,y+sy/2),local(x-sx/2,y+sy/2)]
            # Match only the road-facing surface, never the back of a thin wall.
            face={'outer_left':1,'outer_right':3,'outer_top':0,
                  'inner_left':3,'inner_right':1,'inner_top':2,
                  'end_A':2,'end_D':2}[m.get('name')]
            for a,b in [(corners[face],corners[(face+1)%4])]:
                axis=0 if abs(a[0]-b[0])<1e-6 else 1
                other=1-axis
                self.segments.append((axis,(a[axis]+b[axis])/2,
                                      min(a[other],b[other]),max(a[other],b[other])))
        self.pose=None;self.last_raw=None;self.quality={}

    def predict(self, raw):
        if self.last_raw is None:self.pose=tuple(raw)
        else:
            dx,dy=raw[0]-self.last_raw[0],raw[1]-self.last_raw[1]
            correction=self.pose[2]-self.last_raw[2]
            c,s=math.cos(correction),math.sin(correction)
            self.pose=(self.pose[0]+c*dx-s*dy,self.pose[1]+s*dx+c*dy,
                       wrap(self.pose[2]+wrap(raw[2]-self.last_raw[2])))
        self.last_raw=raw
        return self.pose

    def correct(self, points):
        source=np.asarray(points[::4],dtype=float)
        if len(source)<100:return False
        pose=np.array(self.pose);prior=pose.copy()
        for _ in range(20):
            c,s=math.cos(pose[2]),math.sin(pose[2])
            rotated=source@np.array([[c,s],[-s,c]])
            xy=rotated+pose[:2]
            best=np.full(len(source),np.inf);axes=np.zeros(len(source),dtype=int)
            values=np.zeros(len(source))
            for axis,value,lo,hi in self.segments:
                perpendicular=xy[:,axis]-value
                outside=np.maximum(np.maximum(lo-xy[:,1-axis],xy[:,1-axis]-hi),0)
                distance=np.hypot(perpendicular,outside)
                use=(distance<best)&(outside<.025)
                best[use]=distance[use];axes[use]=axis;values[use]=value
            keep=best<.4
            if np.count_nonzero(keep)<100:return False
            indices=np.nonzero(keep)[0];axis=axes[keep]
            residual=xy[indices,axis]-values[keep]
            jac=np.zeros((len(indices),3));jac[np.arange(len(indices)),axis]=1
            jac[:,2]=np.where(axis==0,-rotated[indices,1],rotated[indices,0])
            weights=np.minimum(1,.04/np.maximum(np.abs(residual),1e-6))
            update,_,rank,_=np.linalg.lstsq(jac*weights[:,None],-residual*weights,rcond=None)
            if rank<3:return False
            pose+=update;pose[2]=wrap(pose[2])
            if np.linalg.norm(update)<1e-5:break
        rms=float(np.sqrt(np.mean(residual**2)))
        self.quality=dict(inliers=len(indices),rms_m=rms,
                          correction_m=float(np.linalg.norm(pose[:2]-prior[:2])),
                          correction_yaw=float(wrap(pose[2]-prior[2])))
        if rms>.08 or self.quality['correction_m']>.18 or abs(self.quality['correction_yaw'])>.1:
            return False
        self.pose=tuple(float(v) for v in pose)
        return True
