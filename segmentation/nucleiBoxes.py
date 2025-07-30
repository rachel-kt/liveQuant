
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:18:14 2023

@author: rachel

"""

from skimage.measure import label, regionprops
from skimage import io, data
from dask.array.image import imread
import numpy as np
import math
import os
import tifffile
from cellpose.utils import remove_edge_masks
import numpy.ma as ma
import matplotlib.pyplot as plt

def getNucleiCoordinates(maskPath, shouldIplot=False):
    """Extract nuclei coordinates and related information from a binary mask.

    Parameters
    ----------
    maskPath : str
        Path to the binary mask image containing nuclei.
    shouldIplot : bool, optional
        Flag indicating whether to plot the nuclei and related information, default is False.

    Returns
    -------
    tuple
        A tuple containing the following lists:
        1. cropBoxCoordinates : list
            List of crop box coordinates for each detected nucleus.
        2. nucleiCentroids : list
            List of centroids for each detected nucleus.
        3. noNuclei : ndarray
            Array of unique nucleus labels.
        4. orientations : list
            List of orientation information for each detected nucleus.

    Notes
    -----
    This function reads a binary mask image, labels connected components, and extracts
    information such as crop box coordinates, centroids, nucleus labels, and orientations.
    If shouldIplot is set to True, it also plots the nuclei and related information.

    Example
    -------
    crop_boxes, centroids, nucleus_labels, orientations = getNucleiCoordinates('/path/to/mask/image.tif', shouldIplot=True)
    """
    # Function implementation...
    # ...

    image = io.imread(maskPath)
    label_img = label(image)
    label_img = remove_edge_masks(label_img, change_index=True)
    regions = regionprops(label_img)
    noNuclei = np.unique(label_img)
    noNuclei = np.delete(noNuclei,0)
    if shouldIplot==True:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
    cropBoxCoordinates = []
    nucleiCentroids = []
    orientations = []
    kk=0
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        if shouldIplot==True:           
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=.7)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=.7)
            ax.plot(x0, y0, '.g', markersize=15)  
        minr_, minc_, maxr_, maxc_ = props.bbox
        maxcc = np.max([abs(minr_-maxr_),abs(minc_-maxc_)])
        minr = minr_-0.1*maxcc
        minc = minc_-0.1*maxcc
        maxr = maxr_+0.1*maxcc
        maxc = maxc_+0.1*maxcc
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if shouldIplot==True:            
            ax.plot(bx, by, '-b', linewidth=.7)
            ax.text(x0,y0,noNuclei[kk], color='white')

        nucleiCentroids.append([y0,x0])
        cropBoxCoordinates.append([bx,by])
        orientations.append([x1,y1,x2,y2])
        kk=kk+1
    if shouldIplot==True: 
        ax.axis((0, 1024, 1024, 0))
    plt.show()
    return cropBoxCoordinates, nucleiCentroids, noNuclei, orientations

def getBackgroundTimeProfile(imagePath, nameKey, imsQ, minc, minr, maxc, maxr, start=0, stop=0, extensionF='.tif', saveCrop=True):
    """Calculate the background time profile based on image stack projections.

    Parameters
    ----------
    path_input : str
        Path to the directory containing the image stack.
    nameKey : str
        Key identifying the image stack.
    imsQ : str
        Identifier for the image stack.
    minc : int
        Minimum column index for cropping the image.
    minr : int
        Minimum row index for cropping the image.
    maxc : int
        Maximum column index for cropping the image.
    maxr : int
        Maximum row index for cropping the image.
    start : int, optional
        Starting time point for profile calculation, default is 0.
    stop : int, optional
        Ending time point for profile calculation, default is 0 (single time point).
    extensionF : str, optional
        File extension for image stack files, default is '.tif'.

    Returns
    -------
    list
        Mean intensity values of the cropped region for each time point in the specified range.

    Notes
    -----
    This function reads image stacks, extracts the projection, and calculates the mean intensity
    of the specified crop region for each time point in the given range.

    Example
    -------
    mean_profile = getBackgroundTimeProfile('/path/to/images', 'nuclei_', 'Q1', 10, 20, 30, 40, start=0, stop=10)
    """
    # Function implementation...
    # ...


    meanofRandomImageSample = []
    nucleiStackName = nameKey+imsQ
    nucleiStackPath = os.path.join(imagePath, nucleiStackName+'*.tif') 
    newimageFull = imread(nucleiStackPath)
    bgCropFull = []
    for timePoint in range(start, stop):
        if timePoint%50==0:
            print(timePoint)
        newimage = newimageFull[timePoint]
        imageShape = np.shape(newimage)

        projectionNuclei = np.max(newimage, axis=0)
        cropBoxForIntensity = projectionNuclei[minr:maxr,minc:maxc]
        bgCropFull.append(cropBoxForIntensity)
        meanofRandomImageSample.append(np.mean(cropBoxForIntensity))
    if saveCrop==True:
        print('Saving Crops!')
        bgFileName = os.path.join(imagePath,'background','background_movie'+extensionF)
        bgfolder = os.path.join(imagePath,'background')
        if not os.path.exists(bgfolder):
            os.makedirs(bgfolder)
        with tifffile.TiffWriter(bgFileName, imagej=True) as tif:
            tif.write(np.array(bgCropFull))
        print('\nDone!')
    return meanofRandomImageSample

def getTimeProfile(path_input,nucleiStackForm, cellNumber, label_image_name, labeldf, start=0, stop=0, extensionF='.tif'):
    """Calculate time profiles of mean intensity inside and outside a specified nucleus.

    Parameters
    ----------
    path_input : str
        Path to the directory containing the image stack.
    nucleiStackForm : str
        Prefix identifying the image stack.
    cellNumber : int
        Identifier for the nucleus.
    label_image_name : str
        Path to the labeled image containing nuclei.
    labeldf : pandas.DataFrame
        DataFrame containing information about labeled nuclei.
    start : int, optional
        Starting time point for profile calculation, default is 0.
    stop : int, optional
        Ending time point for profile calculation, default is 0 (single time point).
    extensionF : str, optional
        File extension for image stack files, default is '.tif'.

    Returns
    -------
    tuple
        A tuple containing two lists:
        1. meanofRandomImageSample_within : list
            Time profile of mean intensity within the specified nucleus.
        2. meanofRandomImageSample_outside : list
            Time profile of mean intensity outside the specified nucleus.

    Notes
    -----
    This function reads image stacks, extracts the projection, and calculates the mean intensity
    within and outside a specified nucleus for each time point in the given range.

    Example
    -------
    profile_within, profile_outside = getTimeProfile('/path/to/images', 'nuclei_', 1, '/path/to/label_image.tif', label_df, start=0, stop=10)
    """
    # Function implementation...
    # ...


    label_image = io.imread(label_image_name)
    label_image = label(label_image)
    label_image = remove_edge_masks(label_image, change_index=True)
    nuclei=np.int64(cellNumber)
    nucIdx = np.where(labeldf['label']==np.int64(nuclei))
    minr = labeldf.loc[nucIdx]['minr'].values[0]
    minc = labeldf.loc[nucIdx]['minc'].values[0]
    sizex = np.int64(labeldf.loc[nucIdx]['sizex'].values[0])
    sizey = np.int64(labeldf.loc[nucIdx]['sizey'].values[0])
    nucleiMask = label_image[math.floor(minr):math.floor(minr)+sizex,math.floor(minc):math.floor(minc)+sizey]

    imagePath = path_input
    meanofRandomImageSample_within = []
    meanofRandomImageSample_outside = []
    nucleiStackName = nucleiStackForm+cellNumber
    nucleiStackPath = os.path.join(imagePath, nucleiStackName+'*.tif') 
    newimageFull = imread(nucleiStackPath)

    for timePoint in range(start,stop):

        newimage = newimageFull[timePoint]
        projectionNuclei = np.max(newimage, axis=0)

        withinNuc = ma.masked_where(nucleiMask!=nuclei, projectionNuclei)
        outsideNuc =  ma.masked_where(nucleiMask==nuclei, projectionNuclei)
        meanofRandomImageSample_within.append(np.mean(withinNuc))
        meanofRandomImageSample_outside.append(np.mean(outsideNuc))

    return meanofRandomImageSample_within, meanofRandomImageSample_outside

