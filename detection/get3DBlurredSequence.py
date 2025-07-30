# -*- coding: utf-8 -*-
"""
Created on Tue Jul 3 14:59:14 2023

@author: rachel

This function is for reading cell z stacks crops and running bigfish spot and cluster detection on each movie frame. 
"""

import numpy as np

def get3DBlurredSequence(sequenceCell, windowSize=5):
    """Generate a 3D-blurred sequence from a given cell sequence.

    Parameters
    ----------
    sequenceCell : ndarray
        4D numpy array representing a sequence of images from a cell.
    windowSize : int, optional
        Size of the window for blurring, default is 5.

    Returns
    -------
    ndarray
        3D-blurred sequence generated from the input sequence.

    Notes
    -----
    This function creates a 3D-blurred sequence by applying a mean filter along the time axis
    using a specified window size.

    Example
    -------
    blurred_sequence = get3DBlurredSequence(cell_sequence, windowSize=7)
    
    """
    blurImage = []  
    MaxTimePoint = sequenceCell.shape[0]
    start=windowSize//2
    stop=MaxTimePoint-start
    for t in range(start,stop):
        images = []
        for ii in np.arange(-(windowSize-1)//2,(windowSize+1)//2,1):
            images.append(sequenceCell[t+ii])

        meanImage = np.mean(images, axis=0)
        blurImage.append(np.max(meanImage,axis=0))
    blurImage = np.array(blurImage).astype(np.uint16)
    return blurImage