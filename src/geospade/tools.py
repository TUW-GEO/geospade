import ogr
import shapely
import numpy as np
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
        Distance from the origin to the new point
    angle : float
        Azimuth angle to the new point in respect to the x-axis (horizontal).
    deg : boolean, optional
        Denotes, whether `angle` is being parsed in degrees or radians:
            True    => degrees (default)
            False   => radians

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
    deg: boolean, optional
        Denotes, whether the angles are returned in degrees or radians:
            True    => degrees (default)
            False   => radians

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
        Machine epsilon (default is 1e-9).

    Returns
    -------
    bool
        True if polygon is rectangular, otherwise False.

    """

    inner_angles = get_inner_angles(polygon, deg=False)
    return all([np.abs(np.pi/2. - inner_angle) <= eps for inner_angle in inner_angles])


def bbox_to_polygon(bbox, sref, segment_size=None):
    """
    Create a polygon geometry from a bounding-box `bbox`, given by
    a set of two points, spanning a polygon area.

    bbox : list of 2-tuples
        List of coordinates representing the rectangle-region-of-interest
        in the format of [(lower-left x, lower-left y),
        (upper-right x, upper-right y)].
    sref : SpatialRef
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
    sref : SpatialRef, optional
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
    sref : SpatialRef, optional
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


def _round_geom_coords(geom):
    """
    'Cleans' the coordinates, so that it has rounded coordinates.

    Parameters
    ----------
    geom : ogr.Geometry
        An OGR geometry.

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
        rlon, rlat, rz = [np.round(lon, decimals=DECIMALS),
                          np.round(lat, decimals=DECIMALS),
                          np.round(z, decimals=DECIMALS)]
        rounded_ring.AddPoint(rlon, rlat, rz)

    geometry_out = ogr.Geometry(ogr.wkbPolygon)
    geometry_out.AddGeometry(rounded_ring)

    return geometry_out
