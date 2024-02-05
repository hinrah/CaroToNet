# ----------------------------------------------------------------------------

# 
#  \file    CAREIIDataset.py
#  \author  hinrich
#  \date    2023-08-08
#
# This is the readable version of CAREIIDataset.py,  which was created to allow a use in MeVisLab without license.
# This code is based on a notebook (https://drive.google.com/file/d/1HTM4fM4WmH1rWN_NsprbLmisdvEPoPgq/view?usp=sharing) provided by the vessel wall
# segmentation challenge (https://vessel-wall-segmentation.grand-challenge.org/Data/)
# # ----------------------------------------------------------------------------

from mevis import *
import numpy as np
from pathlib import Path
from ScriptableMacroModule.ScriptableMacroModule import ScriptableMacroModule
from glob import glob

import os
import xml.etree.ElementTree as ET


class CAREIIDataset(ScriptableMacroModule):
    
    def __init__(self, ctx):
        """
        
        :param ctx: MeVisLab module context.
        """
        super().__init__(ctx)
        self._study_dir = None
        
        self._cases = None
        self._masks = None
        self._inner_contours = None
        self._outer_contours = None
        
    def study_directory_changed(self, e):
        """
        Load a new dataset, including the information provided in the .json file
        :param e: callback info
        :return
        """
              
        # get study directory
        self._study_dir = Path(self._ctx.field('inStudyDirectory').value)
        
        # retrieve image names
        cases = list(map(lambda case: case.parts[-1], self._study_dir.glob('*')))
        cases = sorted(cases, key=lambda f: f.split('.')[0])
        
        # set input options
        self._ctx.field("availableCases").value = ",".join(cases)
        self._cases = cases
        self._ctx.field("selectedCase").value = cases[0]
        
    
    def _generate_cso(self, contour, world_matrix, slice, cso_container):
        cso_points = []
        for point in contour:
          cso_points.append((world_matrix@[point[0]+0.5, point[1]+0.5, slice, 1])[:3])
        cso_container.addClosedSpline(cso_points)
        
        
    def _load(self):
        self._ctx.field("outOuterContours.clear").touch()
        self._ctx.field("outInnerContours.clear").touch()
        
        case_id = self._ctx.field('selectedCase').value
        case_dir = os.path.join(self._study_dir, case_id)
        
        world_matrix = np.array(ctx.field("Info.worldMatrix").value)
        
        arteries = ['L', 'R']
        width = ctx.field("Info.sizeX").value
        height = ctx.field("Info.sizeY").value
        for qvj_file in glob( os.path.join(case_dir, "*", '*.QVJ')):
            qvs_file = os.path.join(os.path.dirname(qvj_file), self.get_qvs_fname(qvj_file))
            qvs_root = ET.parse(qvs_file).getroot()
            
            annotated_slices = self.list_contour_slices(qvs_root)
            
            for slice in annotated_slices:
                lumen_cont = self.get_contour(qvs_root, slice, 'Lumen', height=height, width=width)
                wall_cont = self.get_contour(qvs_root, slice, 'Outer Wall', height=height, width=width)
                
                if lumen_cont is not None and wall_cont is not None:
                  self._generate_cso(lumen_cont, world_matrix, slice, ctx.field("outInnerContours.outCSOList").object())
                  self._generate_cso(wall_cont, world_matrix, slice, ctx.field("outOuterContours.outCSOList").object())

        
    def selectCase(self, e):
        """
        Load a single case
        :param e: callback info
        :return
        """

        # load image
        case = self._ctx.field('selectedCase').value
        self._ctx.field('outImage.source').value = str(self._study_dir / case)
        self._ctx.field('outImage.dplImport').touch()
        self._load()

        return
        
    @staticmethod
    def get_qvs_fname(qvj_path):
        qvs_element = ET.parse(qvj_path).getroot().find('QVAS_Loaded_Series_List').find('QVASSeriesFileName')
        return qvs_element.text

    @staticmethod
    def list_contour_slices(qvs_root):
        """
        :param qvs_root: xml root
        :return: slices with annotations
        """
        avail_slices = []
        image_elements = qvs_root.findall('QVAS_Image')
        for slice_id, element in enumerate(image_elements):
            conts = element.findall('QVAS_Contour')
            if len(conts) > 0:
                avail_slices.append(slice_id)
        return avail_slices
    
    @staticmethod
    def get_contour(qvsroot, slice_id, cont_type, height, width):
        qvas_img = qvsroot.findall('QVAS_Image')
        conts = qvas_img[slice_id].findall('QVAS_Contour')
        pts = None
        for cont_id, cont in enumerate(conts):
            if cont.find('ContourType').text == cont_type:
                pts = cont.find('Contour_Point').findall('Point')
                break
        if pts is not None:
            contours = []
            for p in pts:
                contx = float(p.get('x')) / 512 * width
                conty = float(p.get('y')) / 512 * width - (width-height)/2
                # if current pt is different from last pt, add to contours
                if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                    contours.append([contx, conty])
            return np.array(contours)
