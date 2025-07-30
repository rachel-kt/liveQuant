# -*- coding: utf-8 -*-
"""
Created on Tue Jul 3 14:59:14 2023

@author: rachel

This function is for reading cell z stack crops and reordering them. 

Parameters
----------

1. image : the path to the folder of the cell to be opened. (dir)
               type : str
                
2. frame : default 4. The position where the first frame has to be inserted
               type : int
          

Returns
-------

1. image_Reordered  : Reoredered stack of cell.
               type : list of numpy.array
               

"""

import numpy as np

def reorderZstack(image, frame=4):
    imageShape = np.shape(image)
    
    zlen = imageShape[0]
    xlen = imageShape[1]
    ylen = imageShape[2]
    
    slice_0 = image[0,:,:]
    slice_0 = slice_0.reshape(1,xlen,ylen)
    slice_1 = image[1:frame,:,:]
    slice_2 = image[frame:,:,:]
    image_Reordered = np.concatenate([slice_1,slice_0,slice_2], axis=0)
    return image_Reordered