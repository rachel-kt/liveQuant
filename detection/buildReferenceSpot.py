# -*- coding: utf-8 -*-
"""
Created on Tue Jul 3 14:59:14 2023

@author: rachel

"""

import bigfish.stack as stack
import numpy as np
from bigfish.detection.utils import build_reference_spot
from bigfish.detection.utils import get_object_radius_pixel
from bigfish.detection.utils import convert_spot_coordinates
from bigfish.detection.utils import get_spot_volume
from bigfish.detection.spot_modeling import modelize_spot
from bigfish.detection.spot_modeling import precompute_erf
from bigfish.detection.spot_modeling import gaussian_2d
from bigfish.detection.spot_modeling import _initialize_grid_2d
from bigfish.detection.spot_modeling import gaussian_3d
from bigfish.detection.spot_modeling import _initialize_grid_3d
#print("Big-FISH version: {0}".format(bigfish.__version__))
from copy import deepcopy

def buildReferenceSpotFromImages(images, spots_list, alpha, gamma,voxelSize=(600,121,121), objectSize=(400,202,202)):  
    """Build a reference spot from a sequence of cell images and associated spots.

    Parameters
    ----------
    images : list or ndarray
        Sequence of images representing a cell over time.
    spots_list : list
        List of spot coordinates associated with each image.
    alpha : float
        Parameter influencing the number of spots per candidate region.
    gamma : float
        Parameter for filtering step to denoise the image.
    voxelSize : tuple, optional
        Voxel size in nanometers, default is (600, 121, 121).
    objectSize : tuple, optional
        Object size in nanometers, default is (400, 202, 202).

    Returns
    -------
    ndarray
        Reference spot generated from the input images and spots.

    Notes
    -----
    This function builds a reference spot by denoising and combining a sequence of images
    with associated spot coordinates.

    Example
    -------
    reference_spot = buildReferenceSpotFromImages(images, spots_list, alpha=0.2, gamma=1.0)
    """
    
    # Function implementation...
    # ...
    n=len(images)
    # if one image is provided we enlist it
    if not isinstance(images, list):
        stack.check_array(
            images,
            ndim=[2, 3],
            dtype=[np.uint8, np.uint16, np.float32, np.float64])
        ndim = images.ndim
        images = [images]
        is_list = False
    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(
                image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
        is_list = True
    if isinstance(images, list):
        print('image list found!')
    
    l_reference_spot = []
    for ixx in range(n):
        image = deepcopy(images[ixx])
        spots = spots_list[ixx]
        kernel_size=None


        voxel_size=voxelSize
        spot_radius=objectSize
        alpha=alpha  # alpha impacts the number of spots per candidate region
        #beta=1.2  # beta impacts the number of candidate regions to decompose
        gamma=gamma
        ndim=3
        #image=rna
        # get gaussian kernel to denoise the image
        if kernel_size is None and gamma > 0:
            spot_radius_px = get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=spot_radius,
                ndim=ndim)
            kernel_size = tuple([spot_radius_px_ * gamma
                                 for spot_radius_px_ in spot_radius_px])

        # denoise the image
        if kernel_size is not None:
            #print('yes')
            image_denoised = stack.remove_background_gaussian(
                image=image,
                sigma=kernel_size)

        indices = [i for i in range(spots.shape[0])]
        indices = indices[:min(2000, spots.shape[0])]
        candidate_spots = spots[indices, :]
    #    print('number of candidate spots='+str(candidate_spots.shape[0]))

        # check consistency between parameters
        ndim = image_denoised.ndim
        if ndim != spots.shape[1]:
            raise ValueError("Provided image has {0} dimensions but spots are "
                             "detected in {1} dimensions."
                             .format(ndim, spots.shape[1]))
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError(
                    "'voxel_size' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim))
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError(
                    "'spot_radius' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim))
        else:
            spot_radius = (spot_radius,) * ndim

        # compute radius used to crop spot image
        radius_pixel = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        radius = [np.sqrt(ndim) * r for r in radius_pixel]
        radius = tuple(radius)
        radius_z = np.ceil(radius[0]).astype(np.int64)
        z_shape = radius_z * 2 + 1
        radius_yx = np.ceil(radius[-1]).astype(np.int64)
        yx_shape = radius_yx * 2 + 1

        # collect area around each spot

        for i_spot in range(candidate_spots.shape[0]):

            # get spot coordinates
            spot_z, spot_y, spot_x = candidate_spots[i_spot, :]

            # get the volume of the spot
            image_spot, _, = get_spot_volume(
                image_denoised, spot_z, spot_y, spot_x, radius_z, radius_yx)
            # keep images that are not cropped by the borders
            if image_spot.shape == (z_shape, yx_shape, yx_shape):
                l_reference_spot.append(image_spot)
    
    # if not enough spots are detected
    if len(l_reference_spot) <= 30:
        print("Problem occurs during the computation of a reference "
                      "spot. Not enough (uncropped) spots have been detected.")
    else:
        print("Found "+str(len(l_reference_spot)))
    # project the different spot images
    l_reference_spot = np.stack(l_reference_spot, axis=0)
    alpha_ = alpha * 100
    reference_spot = np.percentile(l_reference_spot, alpha_, axis=0)
    reference_spot = reference_spot.astype(image.dtype)
    
    print("Found "+str(len(l_reference_spot))+" spots, max intensity = "+str((np.max(reference_spot))))
    
    return reference_spot, len(l_reference_spot), np.max(reference_spot)
    
    