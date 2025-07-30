# -*- coding: utf-8 -*-
"""
Created on Tue Jul 3 14:59:14 2023

@author: rachel

"""


import os
import numpy as np
import matplotlib.pyplot as plt
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot
from copy import deepcopy
from reorderStack import reorderZstack
import liveQuant as lv

def getSpotAndClusters(imageSequence,reference_spot, cellnumber=1, startTime=0, stopTime=900, thresholdManual=50, beta=1, gamma=1,numberOfSpots=2, radiusCluster=400, voxelSize=(600,121,121), objectSize=(400,202,202), reorder=False, extensionMov='.tif', showProgress = True):

    """Detect spots and clusters within a cell over a specified time range.

    Parameters
    ----------
    pathTocellCrops : str
        Path to the directory containing cell crop images.
    reference_spot : list or None
        List of reference spots or None if not available.
    cellnumber : int, optional
        Cell identifier, default is 1.
    startTime : int, optional
        Starting time point for analysis, default is 0.
    stopTime : int, optional
        Ending time point for analysis, default is 900.
    thresholdManual : int, optional
        Manual threshold for spot detection, default is 50.
    beta : int, optional
        Beta parameter for spot decomposition, default is 1.
    gamma : int, optional
        Gamma parameter for spot decomposition, default is 1.
    numberOfSpots : int, optional
        Minimum number of spots for clustering, default is 2.
    radiusCluster : int, optional
        Radius for clustering, default is 400.
    voxelSize : tuple, optional
        Voxel size in nanometers, default is (600, 121, 121).
    objectSize : tuple, optional
        Object size in nanometers, default is (400, 202, 202).
    reorder : bool, optional
        Flag indicating whether to reorder the Z-stack, default is False.
    extensionMov : str, optional
        File extension for image stack files, default is '.tif'.

    Returns
    -------
    tuple
        A tuple containing three lists:
        1. spotsFrame : list
            List of detected spots at each time point.
        2. clustersFrames : list
            List of detected clusters at each time point.
        3. ThresholdFrames : list
            List of threshold values used for spot detection at each time point.

    Notes
    -----
    This function reads cell crop images, detects spots and clusters, and provides information
    such as spot coordinates, clusters, and threshold values over the specified time range.

    Example
    -------
    spots, clusters, thresholds = getSpotAndClusters('/path/to/cell/crops', reference_spot, cellnumber=1, startTime=0, stopTime=10)
    """
    # Function implementation...
    # ...

    cell = cellnumber
    spotsFrame = []
    ThresholdFrames = []
    clustersFrames = []
    denseRegions = []

#     path_input = pathTocellCrops
#     movieName = path_input.split('/')[-3]
    
    for t in range(startTime, stopTime):
#         path = os.path.join(path_input, movieName+"_cell_"+str(cell)+'_t'+str(f"{t:03}")+extensionMov)
#         rna = stack.read_image(path)
        rna = np.array(imageSequence[t])
        if reorder:
            rna = reorderZstack(rna,4)
        #rna = normalize(rna,6000)
        rna_mip = stack.maximum_projection(rna)
        
        # spot radius
        spot_radius_px = detection.get_object_radius_pixel(
            voxel_size_nm=voxelSize,#(600, 80, 80), 
            object_radius_nm=objectSize, 
            ndim=3)
        
        # LoG filter
        rna_log = stack.log_filter(rna, sigma=spot_radius_px)

        # local maximum detection
        mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)

        # thresholding
        threshold = detection.automated_threshold_setting(rna_log, mask)
        spots_current, _ = detection.spots_thresholding(rna_log, mask, thresholdManual)

        spotsFrame.append(spots_current)
        ThresholdFrames.append(threshold)
        if showProgress==True: print(t)
        spots_post_decomposition, dense_regions, reference_spot_current = lv.decompose_dense_live(
            image=rna, 
            spots=spots_current,
            reference_spot_previous=reference_spot,
            voxel_size=voxelSize, 
            spot_radius=objectSize, 
           # alpha=alpha,  # alpha impacts the number of spots per candidate region
            beta=beta,  # beta impacts the number of candidate regions to decompose
            gamma=gamma)  # gamma the filtering step to denoise the image
        reference_spot_previous = deepcopy(reference_spot_current)
        
        #clustering
        spots_post_clustering, clusters = detection.detect_clusters(
            spots=spots_post_decomposition, 
            voxel_size=voxelSize, 
            radius=radiusCluster, 
            nb_min_spots=numberOfSpots)
        clustersFrames.append(clusters)
    print('done!')
    return spotsFrame, clustersFrames, ThresholdFrames

def saveSpotsNPZ(spotsFrame, clustersFrames, ThresholdFrames, cellName, pathTocellCrops, reference_spot, threshold, outputstring='_spots_and_clusters'):
    """Save spot and cluster information to a NumPy NPZ file.

    Parameters
    ----------
    spotsFrame : list
        List of detected spots at each time point.
    clustersFrames : list
        List of detected clusters at each time point.
    ThresholdFrames : list
        List of threshold values used for spot detection at each time point.
    cellName : str
        Name or identifier for the cell.
    pathTocellCrops : str
        Path to the directory containing cell crop images.
    reference_spot : list
        List of reference spots.
    threshold : int
        Final threshold selected for detection.

    Returns
    -------
    None

    Notes
    -----
    This function saves the detected spot and cluster information, along with threshold values
    and reference spots, to a NumPy NPZ file for future analysis.

    Example
    -------
    saveSpotsNPZ(spotsFrame, clustersFrames, ThresholdFrames, 'Cell_1', '/path/to/cell/crops', reference_spot)
    """
    # Function implementation...
    # ...
    outfileName = os.path.join(pathTocellCrops,str(cellName)+outputstring)
    np.savez(outfileName, 
             spotsFrame=spotsFrame, 
             clustersFrames=clustersFrames,
             ThresholdFrames=ThresholdFrames,
             reference_spot=reference_spot,
             threshold = threshold,
             allow_pickle=True) 

