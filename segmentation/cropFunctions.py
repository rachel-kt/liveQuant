

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:18:14 2023

@author: rachel
"""

import math
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


def getCropEstimates(bx0, by0, xt, yt):
    """This function is for reading image masks and obtaining coordinates of individual nuclei in a movie frame. 

    Parameters
    ----------

    1. bx0 : crop coordinates(x) of nuclei with centroid from t projection
                   type : list

    2. by0 : crop coordinates(y) of nuclei with centroid from t projection
                   type : list

    3. xt  : x coordinates of nuclei centroid (from t point)
                   type : float

    4. xt  : y coordinates of nuclei centroid (from t point)
                   type : float



    Returns
    -------

    1. bxt : crop coordinates(x) of nuclei with centroid from t point
                   type : list

    2. byt : crop coordinates(y) of nuclei with centroid from t projection
                   type : list

    """

    minc = bx0[0]
    maxc = bx0[1]
    minr = by0[0]
    maxr = by0[2]
    maxccREf = abs(minc-maxc)#np.max([abs(minr-maxr), abs(minc-maxc)])
    maxrrREf = abs(minr-maxr)#np.max([abs(minr-maxr), abs(minc-maxc)])
#     maxccREf = np.max([abs(minr-maxr), abs(minc-maxc)])
#     np.round(maxccREf)//2
    nminc_ = xt-np.round(maxccREf)//2
    nmaxc_ = xt+np.round(maxccREf)//2
    nminr_ = yt-np.round(maxrrREf)//2
    nmaxr_ = yt+np.round(maxrrREf)//2

    bxt = (nminc_, nmaxc_, nmaxc_, nminc_, nminc_)
    byt = (nminr_, nminr_, nmaxr_, nmaxr_, nminr_)
    
    return bxt, byt


def getCentroidAndOrientationImage(label_img):
    """This function is for reading image masks and obtaining coordinates of individual nuclei in a movie frame. 

    Parameters
    ----------

    1. label_img : segmentation mask
                   type : ndarray

    Returns
    -------

    1. coords : list of centroid, orientation, regionprops
                   type : list

    """
    regions = regionprops(label_img)
    coords = []
    for props in regions:
        y0, x0 = props.centroid # centroid
        orientation = props.orientation # orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        coords.append([x0,y0,x1,y1,x2,y2, orientation, props])
    return coords

def makeParameterDf(coordsNuc):
    parameterList = []
    for kk in range(len(coordsNuc)):
        props = coordsNuc[kk][-1]
        y00, x00 = props.centroid   
        minr_, minc_, maxr_, maxc_ = props.bbox
        maxcc = abs(minc_-maxc_)#np.max([abs(minr_-maxr_),abs(minc_-maxc_)])
        maxrr = abs(minr_-maxr_)
        minr = minr_-0.01*maxrr
        minc = minc_-0.01*maxcc
        maxr = maxr_+0.01*maxrr
        maxc = maxc_+0.01*maxcc
        parameterList.append([y00,x00, props.label,minr, minc, maxr, maxc, maxcc, np.abs(y00-x00)])

    parameterList = np.array(parameterList)
    parameterListDF = pd.DataFrame(parameterList[np.argsort(parameterList[:,8]),:], columns=['x','y','label', 'minr', 'minc','maxr', 'maxc', 'maxRef', 'diff'])
    return parameterListDF
