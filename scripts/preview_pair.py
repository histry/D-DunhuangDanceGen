"""Render synchronized measured Native FK (not a mesh or a quality judgment)."""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('anchored',type=Path);p.add_argument('output',type=Path);p.add_argument('--audio',type=Path);a=p.parse_args()
parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
tracks=[np.load(d/'native_fk.npy') for d in (a.baseline,a.anchored)]
if tracks[0].shape!=tracks[1].shape or tracks[0].shape[1:]!=(24,3):raise ValueError('Two matched SMPL24 tracks required')
frames=len(tracks[0]);anchors={x['frame'] for x in json.loads((a.anchored/'request.json').read_text())['anchors']}
# Same orthographic camera and scale, each dancer centered on its initial root XZ.
projected=[]
for track in tracks:
    t=track.copy();t[:,:,[0,2]]-=t[0,0,[0,2]].copy();t[:,:,1]-=np.percentile(t[:,[7,8,10,11],1],5)
    projected.append(np.stack([.94*t[:,:,0]+.342*t[:,:,2],t[:,:,1]],axis=-1))
allcoords=np.stack(projected);scale=min(435/max(abs(allcoords[...,0]).max()*2,.001),330/max(np.ptp(allcoords[...,1]),.001))
low=float(allcoords[...,1].min());height,width=480,960
cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','960x480','-r','30','-i','-']
if a.audio:cmd+=['-i',str(a.audio),'-map','0:v','-map','1:a','-c:a','aac','-shortest']
cmd+=['-frames:v',str(frames),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
for f in range(frames):
    im=np.full((height,width,3),248,np.uint8)
    for k,xy in enumerate(projected):
        mid=240+480*k;pts=np.rint(np.stack([mid+xy[f,:,0]*scale,414-(xy[f,:,1]-low)*scale],axis=1)).astype(int)
        color=(150,87,43) if k==0 else (70,110,20)
        cv2.putText(im,[a.left_label,a.right_label][k],(24+k*480,40),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
        y=int(round(414+low*scale));cv2.line(im,(14+k*480,y),(466+k*480,y),(205,205,205),1)
        for j,parent in enumerate(parents[1:],1):cv2.line(im,tuple(pts[parent]),tuple(pts[j]),color,3,cv2.LINE_AA)
        for j,point in enumerate(pts):cv2.circle(im,tuple(point),4,(20,70,220) if k==1 and f in anchors and 12<=j<20 else color,-1,cv2.LINE_AA)
    cv2.line(im,(480,10),(480,425),(210,210,210),1)
    cv2.putText(im,'Native FK | %.2f s | %d / %d frames'%(f/30,f+1,frames),(24,457),cv2.FONT_HERSHEY_SIMPLEX,.54,(55,55,55),1,cv2.LINE_AA)
    proc.stdin.write(im.tobytes())
proc.stdin.close()
if proc.wait():raise SystemExit('ffmpeg failed')
