# -*- coding: utf-8 -*-
# License: BSD 3 clause

"""
The bigfish.detection subpackage includes function to detect RNA spot in 2-d
and 3-d.
"""

from bigfish.detection.spot_detection import detect_spots
from bigfish.detection.spot_detection import local_maximum_detection
from bigfish.detection.spot_detection import spots_thresholding
from bigfish.detection.spot_detection import automated_threshold_setting
from bigfish.detection.spot_detection import get_elbow_values

from bigfish.detection.dense_decomposition import decompose_dense
from bigfish.detection.dense_decomposition import get_dense_region
from bigfish.detection.dense_decomposition import simulate_gaussian_mixture
from .dense_decomposition_live_cell import decompose_dense_live, get_dense_region_live

from bigfish.detection.spot_modeling import modelize_spot
from bigfish.detection.spot_modeling import initialize_grid
from bigfish.detection.spot_modeling import gaussian_2d
from bigfish.detection.spot_modeling import gaussian_3d
from bigfish.detection.spot_modeling import precompute_erf
from bigfish.detection.spot_modeling import fit_subpixel

from bigfish.detection.cluster_detection import detect_clusters

from bigfish.detection.utils import convert_spot_coordinates
from bigfish.detection.utils import get_object_radius_pixel
from bigfish.detection.utils import get_object_radius_nm
from bigfish.detection.utils import build_reference_spot
from bigfish.detection.utils import get_spot_volume
from bigfish.detection.utils import get_spot_surface
from bigfish.detection.utils import compute_snr_spots
from bigfish.detection.utils import get_breaking_point


_spots = [
    "detect_spots",
    "local_maximum_detection",
    "spots_thresholding",
    "automated_threshold_setting",
    "get_elbow_values"]

_dense = [
    "decompose_dense",
    "get_dense_region",
    "simulate_gaussian_mixture"]

_model = [
    "modelize_spot",
    "initialize_grid",
    "gaussian_2d",
    "gaussian_3d",
    "precompute_erf",
    "fit_subpixel"]

_clusters = [
    "detect_clusters"]

_utils = [
    "convert_spot_coordinates",
    "get_object_radius_pixel",
    "get_object_radius_nm",
    "build_reference_spot",
    "get_spot_volume",
    "get_spot_surface",
    "compute_snr_spots",
    "get_breaking_point"]

__all__ = _spots + _dense + _model + _clusters + _utils
