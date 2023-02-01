""" Module collecting all functions dealing with coordinate transformations. """

import warnings
import numpy as np
from osgeo import osr
from osgeo import ogr
from typing import Tuple
from geospade import DECIMALS
from geospade.tools import _round_geom_coords


def xy2ij(x, y, geotrans, origin="ul") -> Tuple[int or np.ndarray, int or np.ndarray]:
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    x : float or np.array
        World system coordinate(s) in X direction.
    y : float or np.array
        World system coordinate(s) in Y direction.
    geotrans : 6-tuple
        GDAL geo-transformation parameters/dictionary.
    origin : str, optional
        Defines the world system origin of the pixel. It can be:
        - upper left ("ul", default)
        - upper right ("ur")
        - lower right ("lr")
        - lower left ("ll")
        - center ("c")

    Returns
    -------
    i : int or np.ndarray
        Column number(s) in pixels.
    j : int or np.ndarray
        Row number(s) in pixels.

    """

    px_shift_map = {"ul": (0, 0),
                    "ur": (1, 0),
                    "lr": (1, 1),
                    "ll": (0, 1),
                    "c": (.5, .5)}

    px_shift = px_shift_map.get(origin, None)
    if px_shift is None:
        wrng_msg = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead.".format(origin)
        warnings.warn(wrng_msg)
        px_shift = (0, 0)

    # shift world system coordinates to the desired pixel origin
    x -= px_shift[0] * geotrans[1]
    y -= px_shift[1] * geotrans[5]

    # solved equation system describing an affine model: https://gdal.org/user/raster_data_model.html
    i = np.around((-1.0 * (geotrans[2] * geotrans[3] - geotrans[0] * geotrans[5] + geotrans[5] * x - geotrans[2] * y)/
                   (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])), decimals=DECIMALS).astype(int)
    j = np.around((-1.0 * (-1 * geotrans[1] * geotrans[3] + geotrans[0] * geotrans[4] - geotrans[4] * x + geotrans[1] * y)/
                   (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])), decimals=DECIMALS).astype(int)

    return i, j


def ij2xy(i, j, geotrans, origin="ul") -> Tuple[float or np.ndarray, float or np.ndarray]:
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    i : int or np.array
        Column number(s) in pixels.
    j : int or np.array
        Row number(s) in pixels.
    geotrans : 6-tuple
        GDAL geo-transformation parameters/dictionary.
    origin : str, optional
        Defines the world system origin of the pixel. It can be:
        - upper left ("ul", default)
        - upper right ("ur")
        - lower right ("lr")
        - lower left ("ll")
        - center ("c")

    Returns
    -------
    x : float or np.ndarray
        World system coordinate(s) in X direction.
    y : float or np.ndarray
        World system coordinate(s) in Y direction.

    """

    px_shift_map = {"ul": (0, 0),
                    "ur": (1, 0),
                    "lr": (1, 1),
                    "ll": (0, 1),
                    "c": (.5, .5)}

    px_shift = px_shift_map.get(origin, None)
    if px_shift is None:
        wrng_msg = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead".format(origin)
        warnings.warn(wrng_msg)
        px_shift = (0, 0)

    # shift pixel coordinates to the desired pixel origin
    i += px_shift[0]
    j += px_shift[1]

    # applying affine model: https://gdal.org/user/raster_data_model.html
    x = geotrans[0] + i * geotrans[1] + j * geotrans[2]
    y = geotrans[3] + i * geotrans[4] + j * geotrans[5]

    return x, y


def transform_coords(x, y, this_sref, other_sref) -> Tuple[float, float]:
    """
    Transforms coordinates from a source to a target spatial reference system.

    Parameters
    ----------
    x : float
        World system coordinate in X direction with `this_sref` as a spatial reference system.
    y : float
        World system coordinate in Y direction with `this_sref` as a spatial reference system.
    this_sref : geospade.crs.SpatialRef, optional
        Spatial reference of the source coordinates.
    other_sref : geospade.crs.SpatialRef, optional
        Spatial reference of the target coordinates.

    Returns
    -------
    x : float
        World system coordinate in X direction with `other_sref` as a spatial reference system.
    y : float
        World system coordinate in Y direction with `other_sref` as a spatial reference system.

    """

    ct = osr.CoordinateTransformation(this_sref.osr_sref, other_sref.osr_sref)
    x, y, _ = ct.TransformPoint(x, y)

    return x, y


def transform_geom(geom, other_sref) -> ogr.Geometry:
    """
    Transforms a OGR geometry from its source to a target spatial reference system.

    Parameters
    ----------
    geom : ogr.Geometry
        OGR geometry with an assigned spatial reference system.
    other_sref : geospade.crs.SpatialRef, optional
        Spatial reference of the target geometry.

    Returns
    -------
    geom : ogr.Geometry
        Transformed OGR geometry.

    """

    geometry_out = geom.Clone()
    # transform geometry to new spatial reference system.
    geometry_out.TransformTo(other_sref.osr_sref)
    # assign new spatial reference system
    geometry_out.AssignSpatialReference(other_sref.osr_sref)

    return geometry_out


def build_geotransform(ul_x, ul_y, x_pixel_size, y_pixel_size, rot, deg=True) -> Tuple[float, float, float,
                                                                                       float, float, float]:
    """
    A helper function, that constructs the GDAL geo-transformation tuple given the
    upper-left coordinates, the pixel sizes and the rotation angle with respect
    to the world system grid.

    Parameters
    ----------
    ul_x : float
        X coordinate of the upper-left corner.
    ul_y : float
        Y coordinate of the upper-left corner.
    x_pixel_size : float
        Absolute pixel size in X direction.
    y_pixel_size : float
        Absolute pixel size in Y direction.
    rot : float
        Rotation angle in  degrees and radians depending on `deg`.
    deg : boolean
        Denotes, whether `angle` is being parsed in degrees or radians:
            - True => degrees (default)
            - False => radians

    Returns
    -------
    6-tuple
        GDAL geo-transformation tuple.

    """

    gt = [0, 1, 0, 0, 0, 1]
    gt[0], gt[3] = ul_x, ul_y
    alpha = np.radians(rot) if deg else rot
    gt[1] = np.cos(alpha) * x_pixel_size
    gt[2] = -np.sin(alpha) * x_pixel_size
    gt[4] = np.sin(alpha) * y_pixel_size
    gt[5] = np.cos(alpha) * y_pixel_size

    return tuple(gt)
