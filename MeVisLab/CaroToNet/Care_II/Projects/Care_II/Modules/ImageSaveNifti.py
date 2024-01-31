# ----------------------------------------------------------------------------

# 
#  \file    ImageSaveNifti.py
#  \author  hinrich
#  \date    2023-06-08
#
#  

# ----------------------------------------------------------------------------

from mevis import MLABFileDialog
import os
import nibabel as nib
import numpy as np

formatExtensions = {
  "NIFTI" : [".nii", "nii.gz"],
}

def browse():
  format = "NIFTI"
  formatFilter = "{} ({})".format(format, " ".join("*{}".format(f) for f in formatExtensions[format]))
  ctx.field("inDestination").value = MLABFileDialog.getSaveFileName("", formatFilter, "Select file")
  save()
    
    

def save():
  flip_x_y = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]])
  
  translate_to_edge = np.array([[1,0,0,0.5], [0,1,0,0.5], [0,0,1,0.5], [0,0,0,1]])
  
  destination = ctx.field("inDestination").value
  voxel_to_world_matrix = ctx.field("OriginalImage.output0").voxelToWorldMatrix()

  img = ctx.field("OriginalImage.output0").image()
  extent = ctx.field("OriginalImage.output0").size6D()
  tile = img.getTile((0, 0, 0, 0, 0, 0), extent).transpose()
  
  niftiImage = nib.Nifti1Image(tile, flip_x_y@voxel_to_world_matrix@translate_to_edge)
  nib.save(niftiImage, destination)