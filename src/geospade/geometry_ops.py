import math

def construct_geotransform(orig, rot, px, deg=True):
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
