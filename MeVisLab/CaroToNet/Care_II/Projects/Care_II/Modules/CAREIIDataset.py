from mevis import *
from numpy import array as a
from pathlib import Path
from ScriptableMacroModule.ScriptableMacroModule import ScriptableMacroModule
from glob import glob
import os.path as p
import xml.etree.ElementTree as ET
class CAREIIDataset(ScriptableMacroModule):
 def s_d_c(self, e):
  self._sd=Path(ctx.field('inStudyDirectory').value)
  ctx.field("availableCases").value=",".join(list(map(lambda c: c.parts[-1], self._sd.glob('*'))))
 def g_cs(_, c, wm, s, cso):
  cso.addClosedSpline([(wm@[p[0]+0.5, p[1]+0.5, s, 1])[:3] for p in c])
 def s_c(self, e):
  ctx.field('outImage.source').value=str(self._sd/ctx.field('selectedCase').value)
  ctx.field('outImage.dplImport').touch()
  ctx.field("outOuterContours.clear").touch()
  ctx.field("outInnerContours.clear").touch()
  cdir=p.join(self._sd, ctx.field('selectedCase').value)
  wm=a(ctx.field("Info.worldMatrix").value)
  w=ctx.field("Info.sizeX").value
  h=ctx.field("Info.sizeY").value
  for qvj in glob(p.join(cdir,"*",'*.QVJ')):
   qvs=p.join(p.dirname(qvj),ET.parse(qvj).getroot().find('QVAS_Loaded_Series_List').find('QVASSeriesFileName').text)
   qvsr=ET.parse(qvs).getroot()
   for s in self.l_c_s(qvsr):
    l_c=self.g_c(qvsr,s,'Lumen',h=h,w=w)
    w_c=self.g_c(qvsr,s,'Outer Wall',h=h,w=w)
    if l_c is not None and w_c is not None:
     self.g_cs(l_c,wm,s,ctx.field("outInnerContours.outCSOList").object())
     self.g_cs(w_c,wm,s,ctx.field("outOuterContours.outCSOList").object())
 def l_c_s(_,r):
  a=[]
  for si, e in enumerate(r.findall('QVAS_Image')):
   if e.findall('QVAS_Contour'):
    a.append(si)
  return a
 def g_c(_,r,si,t,h,w):
   conts=r.findall('QVAS_Image')[si].findall('QVAS_Contour')
   pts=None
   for cont in conts:
    if cont.find('ContourType').text == t:
     pts=cont.find('Contour_Point').findall('Point')
     break
   if pts:
    c=[]
    for p in pts:
     nc=[float(p.get('x'))/512*w,float(p.get('y'))/512*w-(w-h)/2]
     if not c or c[-1]!=nc:
       c.append(nc)
    return a(c)