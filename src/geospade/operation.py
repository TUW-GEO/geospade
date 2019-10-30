import math
from numba import jit
import numpy as np
import cv2


def rasterise_polygon(points, sres=1., buffer=0.):
    """
    Rasterises a polygon defined by clockwise list of points with the edge-flag algorithm.


    Parameters
    ----------
    points: list of tuples
        Clockwise list of x and y coordinates defining a polygon.
    sres: float, optional
        Spatial resolution of the raster (default is 1).
    inner: bool, optional
        If true, only pixels with their center inside the polygon will be marked as a polygon pixel.

    Returns
    -------
    raster: 2D list
        Binary list where zeros are background pixels and ones are foreground (polygon) pixels.

    Notes
    -----
    The edge-flag algorithm was partly taken from https://de.wikipedia.org/wiki/Rasterung_von_Polygonen
    """

    EPS = 1e-6  # machine epsilon, hard-coded
    buffer = abs(buffer)  # TODO: for the time being only the absolute value is used
    # split tuple points into x and y coordinates (Note: zip does not work with Numba)
    xs, ys = list(zip(*points))
    # define extent of the polygon
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # number of columns and rows
    n_rows = round((y_max - y_min)/sres) + 1 + 2*buffer
    n_cols = round((x_max - x_min)/sres) + 1 + 2*buffer
    # raster with zeros
    raster = np.zeros((n_rows, n_cols), np.uint8)

    # first, draw contour of polygon
    for idx in range(1, len(xs)):  # loop over all points of the polygon
        x_1 = xs[idx - 1]
        x_2 = xs[idx]
        y_1 = ys[idx - 1]
        y_2 = ys[idx]
        x_diff = x_2 - x_1
        y_diff = y_2 - y_1
        if y_diff == 0.:  # horizontal line (will be filled later on)
            continue

        k = y_diff / x_diff if x_diff != 0. else None  # slope is None if the line is vertical

        # define start, end and iterator y coordinate and start x coordinate
        if y_1 < y_2:
            y_start = y_1
            y_end = y_2
            x_start = x_1
            y = y_1
        else:
            y_start = y_2
            y_end = y_1
            x_start = x_2
            y = y_2

        while abs(y - y_end - sres) >= EPS:  # iterate along polyline
            if k is not None:
                x = (y - y_start)/k + x_start   # compute x coordinate depending on y coordinate
            else:  # vertical -> x coordinate does not change
                x = x_start

            # compute raster indexes
            i = round(abs((y - y_max)) / sres) + buffer
            j = round(abs((x - x_min)) / sres) + buffer
            raster[i, j] = 1
            y = y + sres  # increase y in steps of 'sres'

    # loop over rows and fill raster from left to right
    for i in range(n_rows):
        is_inner = False
        if sum(raster[i, :]) < 2:  # if there is only one contour point in a line (e.g. spike of a polygon), continue
            continue
        for j in range(n_cols):
            if raster[i, j]:
                is_inner = ~is_inner
            if is_inner:
                raster[i, j] = 1

    if buffer != 0.:
        kernel = np.ones((3, 3), np.uint8)
        raster = cv2.erode(raster, kernel, iterations=buffer)
        raster = raster[buffer:-buffer, buffer:-buffer]

    return raster


@jit(nopython=True)
def get_quadrant(x, y):
    """
    Returns the quadrant in a mathematical positive system:
    1: first quadrant
    2: second quadrant
    3: third quadrant
    4: fourth quadrant
    None: either one or both coordinates are zero

    Parameters
    ----------
    x: float
        x coordinate.
    y: float
        y coordinate.

    Returns
    -------
    int or None
        Quadrant number.
    """

    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    elif x > 0 and y < 0:
        return 4
    else:
        return None


@jit(nopython=True)
def construct_geotransform(ori, rot, px, deg=True):
    """
    Helping function, that constructs the GDAL geotransform tuple given the
    origin, rotation angle in respect to grid and pixel sizes
    :param orig: 2 - list/tuple
        Coordinates of the lower-left corner, x first
    :param rot: float
        Rotation angle in  degrees and radians depending on 'deg' param
    :param px: 2 - list/tuple
        Tuple of pixel sizes, x first
    :param deg: boolean
        Denotes, whether angle is being parsed in degrees or radians
        True    => Degrees (default)
        False   => Radians
    :return: 6 - tuple
        GDAL geotransform tuple
    """
    gt = [0, 1, 0, 0, 0, 1]
    gt[0], gt[3] = orig
    a = round(math.radians(rot)) if deg else rot
    xpx, ypx = px
    gt[1] = math.cos(a) * xpx
    gt[2] = -math.sin(a) * xpx
    gt[4] = math.sin(a) * ypx
    gt[5] = math.cos(a) * ypx
    return tuple(gt)


@jit(nopython=True)
def polar_point(orig, dist, angle):
    """
    Computes a new point by specifying a distance and an azimuth from a given
    known point (Polarpunkt Verfahren).
    :param orig: 2-tuple or 2-list
        tuple or list containing the coordinates of the known point
    :param dist: float
        distance from the known point to the new point
    :param angle: float
        azimuth to the new point in respect to the x-axis (horizontal)
        in radians in interval from -pi to +pi
    :return: tuple
        coordinates of new point
    """
    x, y = orig
    nx = round(x + dist * math.cos(angle),13)
    ny = round(y + dist * math.sin(angle),13)

    return nx, ny
