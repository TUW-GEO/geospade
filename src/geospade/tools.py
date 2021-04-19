"""
The tools module collects general-purpose, geospatial functions for raster and vector geometries and their
interaction.

"""

import ogr
import cv2
import shapely.wkt
import numpy as np
from numba import njit
from copy import deepcopy
from geospade import DECIMALS
from geospade.errors import SrefUnknown
from geospade.errors import GeometryUnknown


def get_quadrant(x, y):
    """
    Returns the quadrant as an interger in a mathematical positive system:
    1 => first quadrant
    2 => second quadrant
    3 => third quadrant
    4 => fourth quadrant
    None => either one or both coordinates are zero

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
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


def polar_point(x_ori, y_ori, dist, angle, deg=True):
    """
    Computes a new point by specifying a distance and an azimuth from a given
    known point. The computation and values refer to a mathematical positive
    system.

    Parameters
    ----------
    x_ori : float
        x coordinate of the origin.
    y_ori :
        y coordinate of the origin.
    dist : float
        Distance from the origin to the new point.
    angle : float
        Azimuth angle to the new point with respect to the x-axis (horizontal).
    deg : boolean, optional
        Denotes, whether `angle` is being parsed in degrees or radians:
            - True    => degrees (default)
            - False   => radians

    Returns
    -------
    x_pp : float
        x coordinate of the new polar point.
    y_pp : float
        y coordinate of the new polar point.

    """
    angle = np.radians(angle) if deg else angle
    x_pp = np.around(x_ori + dist * np.cos(angle), decimals=DECIMALS)
    y_pp = np.around(y_ori + dist * np.sin(angle), decimals=DECIMALS)

    return x_pp, y_pp


def get_inner_angles(polygon, deg=True):
    """
    Computes inner angles between all adjacent poly-lines.

    Parameters
    ----------
    polygon : ogr.Geometry
        Clock-wise ordered OGR polygon.
    deg : boolean, optional
        Denotes, whether the angles are returned in degrees or radians:
            - True    => degrees (default)
            - False   => radians

    Returns
    -------
    inner_angles : list of numbers
        Inner angles in degree or radians.

    """

    polygon = shapely.wkt.loads(polygon.ExportToWkt())

    vertices = list(polygon.exterior.coords)
    vertices.append(vertices[1])
    inner_angles = []
    for i in range(1, len(vertices)-1):
        prev_vertice = np.array(vertices[i-1])
        this_vertice = np.array(vertices[i])
        next_vertice = np.array(vertices[i+1])
        a = prev_vertice - this_vertice
        b = next_vertice - this_vertice
        inner_angle = np.arccos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))
        if deg:
            inner_angle *= (np.pi / 180.)
        inner_angles.append(inner_angle)

    return inner_angles


def is_rectangular(polygon, eps=1e-9):
    """
    Checks if the given polygon is rectangular.

    Parameters
    ----------
    polygon : ogr.Geometry
        Clock-wise ordered OGR polygon.
    eps : float, optional
        Precision at which an angle is considered to be rectangular (default is 1e-9).

    Returns
    -------
    bool
        True if the given polygon is rectangular, otherwise False.

    """

    inner_angles = get_inner_angles(polygon, deg=False)
    return all([np.abs(np.pi/2. - inner_angle) <= eps for inner_angle in inner_angles])


def bbox_to_polygon(bbox, sref, segment_size=None):
    """
    Create a polygon geometry from a bounding-box `bbox`, given by
    a set of two points, spanning a rectangular area.

    bbox : list of 2-tuples
        List of coordinates representing the rectangle-region-of-interest
        in the format of [(lower-left x, lower-left y),
        (upper-right x, upper-right y)].
    sref : geospade.crs.SpatialRef
        Spatial reference system of the coordinates.
    segment_size : float, optional
        For precision: distance of longest segment of the geometry polygon
        in units of the spatial reference system.

    Returns
    -------
    ogr.Geometry
        A polygon geometry representing the input bbox.

    """

    bbox_cp = deepcopy(bbox)

    # wrap around dateline (considering left-lower and right-upper logic).
    if bbox_cp[0][0] > bbox_cp[1][0]:
        bbox_cp[1] = (bbox_cp[1][0] + 360, bbox_cp[1][1])

    # create clock-wise list of corner points
    ll_pt = (float(bbox_cp[0][0]), float(bbox_cp[0][1]))
    ul_pt = (float(bbox_cp[0][0]), float(bbox_cp[1][1]))
    ur_pt = (float(bbox_cp[1][0]), float(bbox_cp[1][1]))
    lr_pt = (float(bbox_cp[1][0]), float(bbox_cp[0][1]))
    corners = [ll_pt, ul_pt, ur_pt, lr_pt]

    return create_polygon_geometry(corners, sref, segment_size=segment_size)


def create_polygon_geometry(points, sref, segment_size=None):
    """
    Creates an OGR polygon geometry defined by a list of points.

    Parameters
    ----------
    points : list of tuples
        Points defining the polygon, either
        2D: [(x1, y1), (x2, y2), ...] or
        3D: [(x1, y1, z1), (x2, y2, z2), ...].
    sref : geospade.crs.SpatialRef, optional
        Spatial reference system of the point coordinates.
    segment_size : float, optional
        For precision: distance of longest segment of the geometry polygon
        in units of the spatial reference system.

    Returns
    -------
    ogr.Geometry
        A polygon defined by the given set of points and located in the
        given spatial reference system.

    """

    # create ring from all points
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in points:
        if len(p) == 2:
            p += (0.0,)
        ring.AddPoint(*p)
    ring.CloseRings()

    # create the geometry
    polygon_geom = ogr.Geometry(ogr.wkbPolygon)
    polygon_geom.AddGeometry(ring)

    # assign spatial reference
    polygon_geom.AssignSpatialReference(sref.osr_sref)

    # modify the geometry such it has no segment longer then the given distance
    if segment_size is not None:
        polygon_geom = segmentize_geometry(polygon_geom, segment_size=segment_size)

    return polygon_geom


def segmentize_geometry(geom, segment_size=1.):
    """
    Segmentizes the lines of a geometry (decreases the point spacing along the lines)
    according to a given `segment_size`.

    Parameters
    ----------
    geom : ogr.Geometry
        OGR geometry object.
    segment_size : float, optional
        For precision: distance of longest segment of the geometry polygon
        in units of the spatial reference system.

    Returns
    -------
    geom_fine : ogr.Geometry
        A congruent geometry realised by more vertices along its shape.

    """

    geom_fine = geom.Clone()
    geom_fine.Segmentize(segment_size)
    geom = None

    return geom_fine


def any_geom2ogr_geom(geom, sref=None):
    """
    Transforms:
        - bounding box extents [(x_min, y_min), (x_max, y_max)]
        - bounding box points [x_min, y_min, x_max, y_max]
        - a list of points [(x_1, y_1), (x_2, y_2), (x_3, y_3), ...]
        - point coordinates (x_1, y_1)
        - a `shapely.geometry.Point` instance
        - a `shapely.geometry.Polygon` instance
        - a `ogr.Geometry` instance
    into an OGR geometry object. If the given geometry representation does not
    contain information about its spatial reference, this information needs to
    be supplied via `sref`.

    Parameters
    ----------
    geom : ogr.Geometry or shapely.geometry or list or tuple
        A vector geometry. It can be a
        - bounding box extent [(x_min, y_min), (x_max, y_max)]
        - bounding box point list [x_min, y_min, x_max, y_max]
        - list of points [(x_1, y_1), (x_2, y_2), (x_3, y_3), ...]
        - point (x_1, y_1)
        - `shapely.geometry.Point` instance
        - `shapely.geometry.Polygon` instance
        - `ogr.Geometry` instance
    sref : geospade.crs.SpatialRef, optional
        Spatial reference system applied to the given geometry if it has none.

    Returns
    -------
    ogr_geom : ogr.Geometry
        Vector geometry as an OGR Geometry object including its spatial reference.

    """

    # a list of two 2-tuples containing the bbox coordinates
    if isinstance(geom, (tuple, list)) and (len(geom) == 2) and isinstance(geom[0], (tuple, list)) \
            and isinstance(geom[1], (tuple, list)):
        ogr_geom = bbox_to_polygon(geom, sref=sref)
    # a list containing 4 coordinates defining the bbox/extent of the geometry
    elif isinstance(geom, (tuple, list)) and (len(geom) == 4) and (all([isinstance(x, (float, int)) for x in geom])):
        bbox_geom = [(geom[0], geom[1]), (geom[2], geom[3])]
        ogr_geom = any_geom2ogr_geom(bbox_geom, sref=sref)
    # a list/tuple with point coordinates
    elif isinstance(geom, (tuple, list)) and (len(geom) == 2) and (all([isinstance(x, (float, int)) for x in geom])):
        point = shapely.geometry.Point(geom[0], geom[1])
        ogr_geom = any_geom2ogr_geom(point)
    # a list containing many 2-tuples
    elif isinstance(geom, (tuple, list)) and (all([isinstance(x, (tuple, list)) and (len(x) == 2) for x in geom])):
        polygon = shapely.geometry.Polygon(geom)
        ogr_geom = any_geom2ogr_geom(polygon)
    # a Shapely point or polygon
    elif isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.Point)):
        ogr_geom = ogr.CreateGeometryFromWkt(geom.wkt)
        ogr_geom = any_geom2ogr_geom(ogr_geom, sref=sref)
    # a OGR geometry
    elif isinstance(geom, ogr.Geometry):
        ogr_geom = geom
        if ogr_geom.GetSpatialReference() is None:
            if sref is None:
                raise SrefUnknown()
            else:
                ogr_geom.AssignSpatialReference(sref.osr_sref)
    else:
        raise GeometryUnknown(geom)

    return ogr_geom


def _round_geom_coords(geom, decimals):
    """
    'Cleans' the coordinates, so that it has rounded coordinates.

    Parameters
    ----------
    geom : ogr.Geometry
        An OGR geometry.
    decimals : int
        Number of significant digits to round to.

    Returns
    -------
    geometry_out : OGRGeometry
        An OGR geometry with 'cleaned' and rounded coordinates.

    """
    if not geom.ExportToWkt().startswith('POLYGON'):
        err_msg = "Only OGR Polygons are supported at the moment."
        raise NotImplementedError(err_msg)

    ring = geom.GetGeometryRef(0)

    rounded_ring = ogr.Geometry(ogr.wkbLinearRing)

    n_points = ring.GetPointCount()

    for p in range(n_points):
        lon, lat, z = ring.GetPoint(p)
        rlon, rlat, rz = [np.round(lon, decimals=decimals),
                          np.round(lat, decimals=decimals),
                          np.round(z, decimals=decimals)]
        rounded_ring.AddPoint(rlon, rlat, rz)

    geometry_out = ogr.Geometry(ogr.wkbPolygon)
    geometry_out.AddGeometry(rounded_ring)

    return geometry_out


@njit
def _fill_raster_poly(raster):
    """
    Fills rasterised polygon.

    Parameters
    ----------
    raster : np.ndarray
        2D, binary numpy array including rasterised polygon boundaries (1=polygon, 0=background).

    Returns
    -------
    raster : np.ndarray
        2D, binary numpy array including rasterised polygon (1=polygon, 0=background).

    """
    n_rows, n_cols = raster.shape
    for i in range(n_rows):
        is_inner = False
        for j in range(n_cols):
            if raster[i, j]:
                is_inner = ~is_inner

            if is_inner:
                raster[i, j] = 1
            else:
                raster[i, j] = 0

    return raster


def rasterise_polygon(geom, x_pixel_size, y_pixel_size, extent=None, buffer=0):
    """
    Rasterises a polygon defined by clockwise list of points with the edge-flag algorithm.

    Parameters
    ----------
    geom : ogr.Geometry
        Clockwise list of x and y coordinates defining a polygon.
    x_pixel_size : float
        Absolute pixel size in X direction.
    y_pixel_size : float
        Absolute pixel size in Y direction.
    extent : 4-tuple, optional
        Output extent of the raster (x_min, y_min, x_max, y_max). If it is not set the output extent is taken from the
        given geometry.
    buffer : int, optional
        Pixel buffer for enlarging the rasterised polygon (default is 0).

    Returns
    -------
    raster : np.array
        Binary array where zeros are background pixels and ones are foreground (polygon) pixels. Its shape is defined by
        the coordinate extent of the input polygon or by the specified `extent` parameter.

    Notes
    -----
    The edge-flag algorithm was partly taken from https://de.wikipedia.org/wiki/Rasterung_von_Polygonen
    The coordinates are always expected to refer to the upper-left corner of a pixel. If the coordinates do not match
    the sampling, they are automatically aligned to upper-left.

    """
    raster_buffer = abs(buffer)

    # retrieve polygon points
    geom_sh = shapely.wkt.loads(geom.ExportToWkt())
    geom_pts = list(geom_sh.exterior.coords)

    # split tuple points into x and y coordinates
    xs, ys = list(zip(*geom_pts))

    # round coordinates to lowest corner
    xs = [int(round(x / x_pixel_size, DECIMALS)) * x_pixel_size for x in xs]
    ys = [int(round(y / y_pixel_size, DECIMALS)) * y_pixel_size for y in ys]

    # define extent of the polygon
    if extent is None:
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    else:
        x_min = int(round(extent[0] / x_pixel_size, DECIMALS)) * x_pixel_size
        x_max = int(round(extent[2] / x_pixel_size, DECIMALS)) * x_pixel_size
        y_min = int(round(extent[1] / y_pixel_size, DECIMALS)) * y_pixel_size
        y_max = int(round(extent[3] / y_pixel_size, DECIMALS)) * y_pixel_size

    # number of columns and rows (+1 to include last pixel row and column, which is lost when computing the difference)
    n_rows = int(round((y_max - y_min) / y_pixel_size, DECIMALS) + 2 * raster_buffer) + 1
    n_cols = int(round((x_max - x_min) / x_pixel_size, DECIMALS) + 2 * raster_buffer) + 1

    # raster with zeros
    raster = np.zeros((n_rows, n_cols), np.uint8)

    # pre-compute half pixel sizes used inside loop
    half_x_pixel_size = np.around(x_pixel_size/2., decimals=DECIMALS)
    half_y_pixel_size = np.around(y_pixel_size/2., decimals=DECIMALS)

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

        k = float(x_diff)/y_diff if x_diff != 0. else None  # slope is None if the line is vertical

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

        while abs(y - y_end) > 10**(-DECIMALS):  # iterate along polyline
            y_s = y + half_y_pixel_size

            if k is not None:
                x_s = np.around((y_s - y_start)*k + x_start, decimals=DECIMALS)   # compute x coordinate depending on y coordinate
            else:  # vertical -> x coordinate does not change
                x_s = x_start

            x_floor = int(x_s/x_pixel_size)*x_pixel_size
            if (x_floor + half_x_pixel_size) <= x_s:
                x_floor += x_pixel_size

            # compute raster indexes
            i = int(round(abs(y_s - y_max) / y_pixel_size, DECIMALS) + raster_buffer)
            j = int(round(abs(x_floor - x_min) / x_pixel_size, DECIMALS) + raster_buffer)
            raster[i, j] = ~raster[i, j]
            y = y + y_pixel_size  # increase y with pixel size

    # loop over rows and fill raster from left to right
    raster = _fill_raster_poly(raster)

    if buffer != 0.:
        kernel = np.ones((3, 3), np.uint8)
        if buffer < 0:
            raster = cv2.erode(raster, kernel, iterations=raster_buffer)
        elif buffer > 0:
            raster = cv2.dilate(raster, kernel, iterations=raster_buffer)

        raster = raster[raster_buffer:-raster_buffer, raster_buffer:-raster_buffer]

    return raster


def rel_extent(origin, extent, x_pixel_size=1, y_pixel_size=1, unit='px'):
    """
    Computes extent in relative pixels or world system coordinates with respect to an origin/reference point for
    the upper left corner.

    Parameters
    ----------
    origin : tuple
        World system coordinates (X, Y) or pixel coordinates (column, row) of the origin/reference point.
    extent : 4-tuple
        Coordinate (min_x, min_y, max_x, max_y) or pixel extent (min_col, min_row, max_col, max_row).
    x_pixel_size : float
        Absolute pixel size in X direction.
    y_pixel_size : float
        Absolute pixel size in Y direction.
    unit : string, optional
        Unit of the relative coordinates.
        Possible values are:
            - 'px':  Relative coordinates are returned as the number of pixels.
            - 'sr':  Relative coordinates are returned in spatial reference units (meters/degrees).

    Returns
    -------
    4-tuple of numbers
        Relative position of the given extent with respect to the given origin.
        The extent values are dependent on the unit:
            - 'px' : (min_col, min_row, max_col, max_row)
            - 'sr' : (min_x, min_y, max_x, max_y)

    """

    rel_extent = (extent[0] - origin[0],
                  extent[1] - origin[1],
                  extent[2] - origin[0],
                  extent[3] - origin[1])
    if unit == 'sr':
        return rel_extent
    elif unit == 'px':
        return np.around(rel_extent[0] / x_pixel_size, decimals=DECIMALS).astype(int),\
               np.around(rel_extent[3] / y_pixel_size, decimals=DECIMALS).astype(int),\
               np.around(rel_extent[2] / x_pixel_size, decimals=DECIMALS).astype(int),\
               np.around(rel_extent[1] / y_pixel_size, decimals=DECIMALS).astype(int)
    else:
        err_msg = "Unit {} is unknown. Please use 'px' or 'sr'.".format(unit)
        raise ValueError(err_msg)