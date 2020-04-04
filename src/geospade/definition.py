import os
import abc
import sys
import osr
import ogr
import copy
import warnings
import shapely
import shapely.wkt
from shapely import affinity

import numpy as np
import geopandas as geopd
import cartopy.crs as ccrs

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from matplotlib.patches import Polygon as PolygonPatch

from geospade.operation import polar_point
from geospade.operation import segmentize_geometry
from geospade.operation import any_geom2ogr_geom
from geospade.operation import construct_geotransform
from geospade.operation import is_rectangular
from geospade.operation import xy2ij
from geospade.operation import ij2xy
from geospade.operation import bbox_to_polygon
from geospade.operation import coordinate_traffo
from geospade.operation import rasterise_polygon

from geospade.spatial_ref import SpatialRef
from geospade import DECIMALS


def _any_geom2ogr_geom(f):
    """
    A decorator which converts an input geometry (first argument) into an `OGR.geometry` object.

    Parameters
    ----------
    f : callable
        Function to wrap around/execute.

    Returns
    -------
    wrapper
        Wrapper around `f`.
    """

    def wrapper(self, *args, **kwargs):
        sref = kwargs.get("sref", self.sref.osr_sref if hasattr(self, "sref") else None)
        if isinstance(sref, SpatialRef):
            sref = sref.osr_sref

        geom = args[0]  # get first argument
        args = args[1:] # remove first argument

        if isinstance(geom, (RasterGeometry, RasterGrid)):
            geom = geom.boundary

        ogr_geom = any_geom2ogr_geom(geom, osr_sref=sref)

        return f(self, ogr_geom, *args, **kwargs)

    return wrapper


class SwathGeometry:
    """
    Represents the geometry of a satellite swath grid.
    """

    def __init__(self, xs, ys, sref, geom_id=None, description=None):
        """
        Constructor of the `SwathGeometry` class.

        Parameters
        ----------
        xs : list of numbers
            East-west coordinates.
        ys : list of numbers
            South-north coordinates.
        sref : geospade.spatial_ref.SpatialRef
            Spatial reference of the geometry.
        geom_id : int or str, optional
            ID of the geometry.
        description : string, optional
            Verbal description of the geometry.
        """
        err_msg = "'SwathGeometry' is not implemented yet."
        raise NotImplementedError(err_msg)

# TODO: add rotation functionality
# TODO: make other -> this geometry projection more efficient
class RasterGeometry:
    """
    Represents the geometry of a georeferenced raster.
    It describes the extent and the grid of the raster along with its spatial reference.
    The (boundary) geometry can be used as an OGR geometry, or can be exported into a Cartopy projection.
    """
    # TODO: add precise description of orientation (mathematic negativ formulation, like it is done in Geotransform)
    def __init__(self, n_rows, n_cols, sref,
                 geotrans=(0, 1, 0, 0, 0, 1),
                 geom_id=None,
                 description=None,
                 segment_size=None,
                 px_origin="ul",
                 parent=None):
        """
        Constructor of the `RasterGeometry` class.

        Parameters
        ----------
        n_rows : int
            Number of pixel rows.
        n_cols : int
            Number of pixel columns.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Instance representing the spatial reference of the geometry.
        geotrans : 6-tuple, optional
            GDAL geotransform tuple.
        geom_id : int or str, optional
            ID of the geometry.
        description : string, optional
            Verbal description of the geometry.
        segment_size : float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
            If None, only the corner points are used for creating the boundary geometry.
        px_origin : str, optional

        parent : RasterGeometry
            Parent `RasterGeometry` instance.
        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.geotrans = geotrans
        self.id = geom_id
        self.description = description
        self.parent = parent
        self._segment_size = segment_size
        self.px_origin = px_origin

        # get internal spatial reference representation from osr object
        if isinstance(sref, osr.SpatialReference):
            sref = SpatialRef.from_osr(sref)
        self.sref = sref

        # compute orientation of rectangle
        self.ori = -np.arctan2(self.geotrans[2], self.geotrans[1])  # radians, usually 0

        # compute boundary geometry
        boundary = Polygon(self.vertices)
        boundary_ogr = ogr.CreateGeometryFromWkt(boundary.wkt)
        boundary_ogr.AssignSpatialReference(self.sref.osr_sref)
        if segment_size is not None:
            self.boundary = segmentize_geometry(boundary_ogr, segment=segment_size)
        else:
            self.boundary = boundary_ogr

    @classmethod
    def from_extent(cls, extent, sref, x_pixel_size, y_pixel_size, **kwargs):
        """
        Creates a `RasterGeometry` object from a given extent (in units of the
        spatial reference) and pixel sizes in both pixel-grid directions
        (pixel sizes determine the resolution).

        Parameters
        ----------
        extent : tuple or list with 4 entries
            Coordinates defining extent from lower left to upper right corner.
            (lower left x, lower left y, upper right x, upper right y)
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Spatial reference of the geometry/extent.
        x_pixel_size : float
            Resolution in x direction.
        y_pixel_size : float
            Resolution in y direction.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`, `segment_size`,
            `px_origin` or `parent`.

        Returns
        -------
        RasterGeometry
            Raster geometry object defined by the given extent and pixel sizes.
        """

        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        # calculate upper-left corner for geotransform
        ul_x, ul_y = polar_point((ll_x, ll_y), height, np.pi / 2.)
        geotrans = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
        # deal negative pixel sizes, hence the absolute value
        n_rows = int(np.ceil(round(abs(height / y_pixel_size), DECIMALS)))
        n_cols = int(np.ceil(round(abs(width / x_pixel_size), DECIMALS)))
        return cls(n_rows, n_cols, sref, geotrans=geotrans, **kwargs)

    @classmethod
    @_any_geom2ogr_geom
    def from_geometry(cls, geom, x_pixel_size, y_pixel_size, sref=None, **kwargs):
        """
        Creates a `RasterGeometry` object from an existing geometry object.
        Since `RasterGeometry` can represent rectangles only, non-rectangular
        shapely objects get converted into its bounding box. Since, e.g. a `Shapely`
        geometry is not georeferenced, the spatial reference has to be
        specified. Moreover, the resolution in both pixel grid directions has to be given.

        Parameters
        ----------
        geom : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Geometry object from which the `RasterGeometry` object should be created.
        x_pixel_size : float
            Resolution in x direction.
        y_pixel_size : float
            Resolution in y direction.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `geom`.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`, `segment_size`,
            `px_origin` or `parent`.

        Returns
        -------
        RasterGeometry
        """

        geom = shapely.wkt.loads(geom.ExportToWkt())
        geom_ch = geom.convex_hull

        if is_rectangular(geom_ch):  # This means the polygon can be described directly as a RasterGeometry
            geom_pts = list(geom.exterior.coords)  # get boundary coordinates
            # separate coordinates
            xs = np.array([geom_pt[0] for geom_pt in geom_pts[:-1]])
            ys = np.array([geom_pt[1] for geom_pt in geom_pts[:-1]])
            # find corner coordinates
            # first, find upper left coordinates (a corner with the highest y coordinate from the two with the
            # smallest x coordinate)
            ul_idx_1 = np.argmin(xs)
            xs_tmp = copy.copy(xs)
            xs_tmp[ul_idx_1] = np.max(xs)
            ul_idx_2 = np.argmin(xs_tmp)
            if ys[ul_idx_1] >= ys[ul_idx_2]:
                ul_idx = ul_idx_1
            else:
                ul_idx = ul_idx_2

            # based on the upper left corner, find the other corner coordinates in a clockwise way
            ul_x = xs[ul_idx]
            ul_y = ys[ul_idx]
            ur_x = xs[ul_idx-1]
            ur_y = ys[ul_idx-1]
            lr_x = xs[ul_idx-2]
            lr_y = ys[ul_idx-2]
            ll_x = xs[ul_idx-3]
            ll_y = ys[ul_idx-3]

            # azimuth of the bottom base of the rectangle = orientation
            # TODO: test different cases!
            rot = np.arctan2(lr_y - ll_y, lr_x - ll_x)
            rot = rot*-1 if rot >= 0. else rot

            # create GDAL geotransform
            geotrans = construct_geotransform((ul_x, ul_y), rot, (x_pixel_size, y_pixel_size), deg=False)

            # define raster properties
            width = round(np.hypot(lr_x - ll_x, lr_y - ll_y), DECIMALS)
            height = round(np.hypot(ur_x - lr_x, ur_y - lr_y), DECIMALS)
            n_rows = int(np.ceil(round(abs(height / y_pixel_size), DECIMALS)))
            n_cols = int(np.ceil(round(abs(width / x_pixel_size), DECIMALS)))

            return RasterGeometry(n_rows, n_cols, sref, geotrans=geotrans, **kwargs)
        else:
            # geom is not a rectangle
            bbox = geom.bounds
            return cls.from_extent(bbox, sref, x_pixel_size, y_pixel_size, **kwargs)

    @classmethod
    def get_common_geometry(cls, raster_geoms, **kwargs):
        """
        Creates a raster geometry, which contains all the given raster geometries given by ˋraster_geomsˋ.

        Parameters
        ----------
        raster_geoms : list of RasterGeometry
            List of `RasterGeometry` objects.

        Returns
        -------
        RasterGeometry
        """

        # first geometry serves as a reference
        sref = raster_geoms[0].sref
        x_pixel_size = raster_geoms[0].x_pixel_size
        y_pixel_size = raster_geoms[0].y_pixel_size

        x_coords = []
        y_coords = []
        for raster_geom in raster_geoms:

            if raster_geom.sref != sref:
                raise ValueError('Geometries have a different spatial reference.')

            if raster_geom.x_pixel_size != x_pixel_size:
                raise ValueError('Geometries have different pixel-sizes in x direction.')

            if raster_geom.y_pixel_size != y_pixel_size:
                raise ValueError('Geometries have different pixel-sizes in y direction.')

            min_x, min_y, max_x, max_y = raster_geom.outer_extent
            x_coords.extend([min_x, max_x])
            y_coords.extend([min_y, max_y])

        extent = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        return cls.from_extent(extent, sref, x_pixel_size, y_pixel_size, **kwargs)

    @property
    def parent_root(self):
        """ RasterGeometry : Finds and returns the root/original parent `RasterGeometry`. """
        raster_geom = self
        while raster_geom.parent is not None:
            raster_geom = raster_geom.parent
        return raster_geom

    @property
    def is_axis_parallel(self):
        """ bool : True if the `RasterGeometry` is not rotated , i.e. it is axis-parallel. """
        return self.ori == 0.

    @property
    def ll_x(self):
        """ float : x coordinate of the lower left corner. """
        x, _ = self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin)
        return x

    @property
    def ll_y(self):
        """ float: y coordinate of the lower left corner. """
        _, y = self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin)
        return y

    @property
    def ul_x(self):
        """ float : x coordinate of the upper left corner. """
        x, _ = self.rc2xy(0, 0, px_origin=self.px_origin)
        return x

    @property
    def ul_y(self):
        """ float: y coordinate of the upper left corner. """
        _, y = self.rc2xy(0, 0, px_origin=self.px_origin)
        return y

    @property
    def ur_x(self):
        """ float : x coordinate of the upper right corner. """
        x, _ = self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin)
        return x

    @property
    def ur_y(self):
        """ float : y coordinate of the upper right corner. """
        _, y = self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin)
        return y

    @property
    def lr_x(self):
        """ float : x coordinate of the upper right corner. """
        x, _ = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin)
        return x

    @property
    def lr_y(self):
        """ float : y coordinate of the upper right corner. """
        _, y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin)
        return y

    @property
    def x_pixel_size(self):
        """ float : Pixel size in x direction. """
        return np.hypot(self.geotrans[1], self.geotrans[2])

    @property
    def y_pixel_size(self):
        # TODO: add description about minus sign
        """ float : Pixel size in y direction. """

        return -np.hypot(self.geotrans[4], self.geotrans[5])

    @property
    def h_pixel_size(self):
        """ float : Pixel size in W-E direction (equal to `x_pixel_size` if the `RasterGeometry` is axis-parallel). """
        return self.x_pixel_size / np.cos(self.ori)

    @property
    def v_pixel_size(self):
        """ float : Pixel size in N-S direction (equal to `y_pixel_size` if the `RasterGeometry` is axis-parallel). """
        return self.y_pixel_size / np.cos(self.ori)

    @property
    def width(self):
        """ float : Width of the raster geometry. """
        return self.n_cols * abs(self.x_pixel_size)

    @property
    def height(self):
        """ float : Height of the raster geometry. """
        return self.n_rows * abs(self.y_pixel_size)

    @property
    def inner_extent(self):
        """ 4-tuple: Extent of the raster geometry with the pixel origins defined by the class
        (min_x, min_y, max_x, max_y). """
        return min([self.ll_x, self.ul_x]), min([self.ll_y, self.lr_y]), \
               max([self.ur_x, self.lr_x]), max([self.ur_y, self.ul_y])

    @property
    def outer_extent(self):
        """ 4-tuple: Outer extent of the raster geometry containing every pixel
        (min_x, min_y, max_x, max_y). """
        ll_x, ll_y = self.rc2xy(self.n_rows - 1, 0, px_origin="ll")
        ur_x, ur_y = self.rc2xy(0, self.n_cols - 1, px_origin="ur")
        lr_x, lr_y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr")
        ul_x, ul_y = self.rc2xy(0, 0, px_origin="ul")
        return min([ll_x, ul_x]), min([ll_y, lr_y]), max([ur_x, lr_x]), max([ur_y, ul_y])

    @property
    def area(self):
        """ float : Area covered by the raster geometry. """
        return self.width * self.height

    @property
    def centre(self):
        """ 2-tuple: Centre defined by the mass centre of the vertices. """
        return shapely.wkt.loads(self.boundary.Centroid().ExportToWkt()).coords[0]

    @property
    def vertices(self):
        """
        4-list of 2-tuples : A tuple containing all corners (convex hull) in the following form (lower left, lower right,
        upper right, upper left).
        """
        ll_x, ll_y = self.rc2xy(self.n_rows - 1, 0, px_origin="ll")
        ur_x, ur_y = self.rc2xy(0, self.n_cols - 1, px_origin="ur")
        lr_x, lr_y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr")
        ul_x, ul_y = self.rc2xy(0, 0, px_origin="ul")
        vertices = [(ll_x, ll_y),
                    (lr_x, lr_y),
                    (ur_x, ur_y),
                    (ul_x, ul_y),
                    (ll_x, ll_y)]
        return vertices

    @property
    def x_coords(self):
        """ list : Returns all coordinates in x direction. """

        if self.is_axis_parallel:
            min_x, _ = self.rc2xy(0, 0)
            max_x, _ = self.rc2xy(0, self.n_cols - 1)
            return np.arange(min_x, max_x + self.x_pixel_size, self.x_pixel_size).tolist()
        else:
            cols = list(range(self.n_cols))
            return [self.rc2xy(0, col)[0] for col in cols]

    @property
    def y_coords(self):
        """ list : Returns all coordinates in y direction. """

        if self.is_axis_parallel:
            _, min_y = self.rc2xy(self.n_rows - 1, 0)
            _, max_y = self.rc2xy(0, 0)
            return np.arange(max_y, min_y + self.y_pixel_size, self.y_pixel_size).tolist()
        else:
            rows = list(range(self.n_rows))
            return [self.rc2xy(row, 0)[1] for row in rows]

    def to_wkt(self):
        """
        Returns Well Known Text (WKT) representation of the boundary of a `RasterGeometry`.

        Returns
        -------
        str
        """

        return self.boundary.ExportToWkt()

    def to_cartopy_crs(self, bounds=None):
        """
        Creates a PROJ4Projection object that can be used as an argument of the
        cartopy `projection` and `transfrom` kwargs. (`PROJ4Projection` is a
        subclass of a `cartopy.crs.Projection` class)

        Parameters
        ----------
        bounds : 4-tuple, optional
            Boundary of the projection (lower left x, upper right x, lower left y, upper right y).
            The default behaviour (None) sets the bounds to the extent of the raster geometry.

        Returns
        -------
        PROJ4Projection
            `PROJ4Projection` instance representing the spatial reference of the raster geometry.
        """

        if bounds is None:
            bounds = self.outer_extent
        return self.sref.to_cartopy_crs(bounds=bounds)

    def to_shapely_geom(self):
        """  shapely.geometry.Polygon : Boundary of the raster geometry represented as a Shapely polygon. """
        return shapely.wkt.loads(self.boundary.ExportToWkt())

    @_any_geom2ogr_geom
    def intersects(self, other, sref=None):
        """
        Evaluates if this `RasterGeometry` instance and another geometry intersect.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to evaluate an intersection with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other`.

        Returns
        -------
        bool
            True if both geometries intersect, false if not.
        """

        other_sref = other.GetSpatialReference()
        if other_sref is not None and not self.sref.osr_sref.IsSame(other_sref):
            other.TransformTo(self.sref.osr_sref)

        return self.boundary.Intersects(other)

    @_any_geom2ogr_geom
    def touches(self, other, sref=None):
        """
        Evaluates if this `RasterGeometry` instance and another geometry touch each other.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to evaluate a touch operation with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other`.

        Returns
        -------
        bool
            True if both geometries touch each other, false if not.
        """

        other_sref = other.GetSpatialReference()
        if other_sref is not None and not self.sref.osr_sref.IsSame(other_sref):
            other.TransformTo(self.sref.osr_sref)

        return self.boundary.Touches(other)

    @_any_geom2ogr_geom
    def within(self, other, sref=None):
        """
        Evaluates if a geometry is in the raster geometry.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to evaluate a within operation with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other`.

        Returns
        -------
        bool
            True if the given geometry is within the raster geometry, false if not.
        """

        other_sref = other.GetSpatialReference()
        if other_sref is not None and not self.sref.osr_sref.IsSame(other_sref):
            other.TransformTo(self.sref.osr_sref)

        return other.Within(self.boundary)

    # TODO: add origin
    @_any_geom2ogr_geom
    def intersection_by_geom(self, other, sref=None, snap_to_grid=True, segment_size=None, inplace=True):
        """
        Computes an intersection figure of two geometries and returns its
        (grid axes-parallel rectangle) bounding box.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other`.
        snap_to_grid : bool, optional
            If true, the computed corners of the intersection are rounded off to
            nearest pixel corner.
        segment_size : float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
            If None, only the corner points are used for creating the boundary geometry.
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        geospade.definition.RasterGeometry
            Raster geometry instance defined by the bounding box of the intersection geometry.
        """

        other_sref = other.GetSpatialReference()
        if other_sref is not None and not self.sref.osr_sref.IsSame(other_sref):
            other.TransformTo(self.sref.osr_sref)

        if not self.intersects(other):
            return None

        intersection = self.boundary.Intersection(other)
        bbox = intersection.GetEnvelope()
        if snap_to_grid:
            ll_px = self.xy2rc(bbox[0], bbox[2], px_origin="ll")
            ur_px = self.xy2rc(bbox[1], bbox[3], px_origin="ur")
            bbox = self.rc2xy(*ll_px, px_origin="ll") + self.rc2xy(*ur_px, px_origin="ur")

        if segment_size is None:
            segment_size = self._segment_size

        intsct_raster_geom = RasterGeometry.from_extent(bbox, sref=self.sref, x_pixel_size=self.x_pixel_size,
                                                        y_pixel_size=self.y_pixel_size, geom_id=self.id,
                                                        description=self.description, segment_size=segment_size,
                                                        parent=self)

        if inplace:
            self._segment_size = segment_size
            self.boundary = intsct_raster_geom.boundary
            self.n_rows = intsct_raster_geom.n_rows
            self.n_cols = intsct_raster_geom.n_cols
            self.geotrans = intsct_raster_geom.geotrans
            return self
        else:
            return intsct_raster_geom

    # TODO: needs to be tested!
    def intersection_by_pixel(self, px_extent, inplace=False):
        """
        Intersects raster geometry with a pixel extent.

        Parameters
        ----------
        px_extent : 4-tuple
            Extent of intersection given in pixels (min_row, min_col, max_row, max_col).
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        geospade.definition.RasterGeometry
            Raster geometry instance defined by the pixel extent.
        """
        min_row, min_col, max_row, max_col = self.crop_px_extent(*px_extent)
        ul_coords = self.rc2xy(min_row, min_col, px_origin="ul")
        ur_coords = self.rc2xy(min_row, max_col, px_origin="ur")
        lr_coords = self.rc2xy(max_row, max_col, px_origin="lr")
        ll_coords = self.rc2xy(max_row, min_col, px_origin="ll")
        coords = [ul_coords, ur_coords, lr_coords, ll_coords, ul_coords]

        return self.intersection_by_geom(coords, sref=self.sref, inplace=inplace)

    # ToDo: needs to be tested!
    @_any_geom2ogr_geom
    def create_mask(self, other, sref=None, buffer=0):
        """
        Creates a mask of which the extent is defined by the raster geometry and the content by the
        given masking geometry.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other`.
            Note: `sref` is used in the decorator `_any_geom2ogr_geom`.
        buffer : int, optional
            Pixel buffer for crop geometry (default is 0).

        Returns
        -------
        numpy.ndarray
            2D mask, where the (`other`) geometry pixels are set to 1 and everything outside the geometry is set to 0.
        """

        other_sref = other.GetSpatialReference()
        if other_sref is not None and not self.sref.osr_sref.IsSame(other_sref):
            other.TransformTo(self.sref.osr_sref)

        mask = np.zeros((self.n_rows, self.n_cols))  # default mask
        if self.intersects(other):
            # intersect raster geometry with other geometry
            shrnk_raster_geom = self.resize(-1, inplace=False)  # resize by -1 px to align intersection coordinates to inner pixels
            intersection = shrnk_raster_geom.boundary.Intersection(other)
            intersection_coords = list(shapely.wkt.loads(intersection.ExportToWkt()).exterior.coords)
            # snap to the grid, i.e. reset the intersection coordinates to center,
            # which are then used by `rasterise_polygon`
            cntr_coords = [shrnk_raster_geom.rc2xy(*shrnk_raster_geom.xy2rc(intersection_coord[0],
                                                                            intersection_coord[1],
                                                                            px_origin=shrnk_raster_geom.px_origin),
                                                   px_origin="c")
                           for intersection_coord in intersection_coords]
            cntr_x_coords, cntr_y_coords = list(zip(*cntr_coords))
            mask_min_x = min(cntr_x_coords)
            mask_max_x = max(cntr_x_coords)
            mask_min_y = min(cntr_y_coords)
            mask_max_y = max(cntr_y_coords)

            # compute mask boundaries, mask and insert it into global mask
            min_row, min_col = self.xy2rc(mask_min_x, mask_max_y, px_origin="c")
            max_row, max_col = self.xy2rc(mask_max_x, mask_min_y, px_origin="c")
            mask_i = rasterise_polygon(cntr_coords, self.x_pixel_size, buffer=buffer)
            mask[min_row:(max_row + 1), min_col:(max_col + 1)] = mask_i  #+1 because max_row/max_col needs to be included

        return mask

    def xy2rc(self, x, y, sref=None, px_origin=None):
        """
        Calculates an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the coordinates.
            Has to be given if the spatial reference is different than the raster geometry.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul", default)
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")

        Returns
        -------
        r : int
            Pixel row number.
        c : int
            Pixel column number.

        Notes
        -----
        Rounds to the closest, lower integer.
        """

        px_origin = self.px_origin if px_origin is None else px_origin

        if sref is not None:
            x, y = coordinate_traffo(x, y, sref, self.sref.osr_sref)

        c, r = xy2ij(x, y, self.geotrans, origin=px_origin)
        return r, c

    def rc2xy(self, r, c, px_origin=None):
        """
        Returns the coordinates of the center or a corner (dependend on ˋoriginˋ) of a pixel specified
        by a row and column number.

        Parameters
        ----------
        r : int
            Pixel row number.
        c : int
            Pixel column number.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul", default)
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")

        Returns
        -------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
        """

        px_origin = self.px_origin if px_origin is None else px_origin
        return ij2xy(c, r, self.geotrans, origin=px_origin)

    def plot(self, ax=None, facecolor='tab:red', edgecolor='black', alpha=1., proj=None, show=False, label_geom=False):
        """
        Plots the boundary of the raster geometry on a map.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        color : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'tab:red').
        border_color : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        alpha : float, optional
            Opacity of the boundary polygon (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes.
        show : bool, optional
            If true, the plot result is shown (default is False).
        label_geom : bool, optional
            If true, the geometry ID is plotted at the center of the raster geometry (default is False).

        Returns
        -------
        matplotlib.pyplot.axes
            Matplotlib axis containing a Cartopy map with the plotted raster geometry boundary.
        """

        if 'matplotlib' in sys.modules:
            import matplotlib.pyplot as plt
        else:
            err_msg = "Module 'matplotlib' is mandatory for plotting a RasterGeometry object."
            raise ImportError(err_msg)

        trafo = self.to_cartopy_crs()
        if ax is None:
            if proj is None:
                proj = ccrs.Mollweide()
            ax = plt.axes(projection=proj)
            ax.set_global()
            ax.gridlines()
            ax.coastlines()

        patch = PolygonPatch(list(self.to_shapely_geom().exterior.coords), facecolor=facecolor, alpha=alpha,
                             transform=trafo, zorder=0, edgecolor=edgecolor)
        ax.add_patch(patch)

        if self.id is not None and label_geom:
            transform = proj._as_mpl_transform(ax)
            ax.annotate(str(self.id), xy=self.centre, xycoords=transform, va="center", ha="center")

        if show:
            plt.show()

        return ax

    def scale(self, scale_factor, segment_size=None, inplace=True):
        """
        Scales the raster geometry as a whole or for each edge.

        Parameters
        ----------
        scale_factor : number or list of numbers
            Scale factors, which have to be given in a clock-wise order, i.e. [left edge, top edge, right edge,
            bottom edge] or as one value.
        segment_size : float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
            If None, only the corner points are used for creating the boundary geometry.
        inplace : bool, optional
            If true, the current instance will be modified (default).
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        geospade.definition.RasterGeometry
            Scaled raster geometry.
        """

        return self.resize(scale_factor, unit='', segment_size=segment_size, inplace=inplace)

    def resize(self, buffer_size, unit='px', segment_size=None, inplace=True):
        """
        Resizes the raster geometry. The value can be specified in
        percent, pixels, or spatial reference units. A positive value extends, a
        negative value shrinks the original object.

        Parameters
        ----------
        buffer_size : number or list of numbers
            Buffering values, which have to be given in a clock-wise order, i.e. [left edge, top edge, right edge,
            bottom edge] or as one value.
        unit: string, optional
            Unit of the buffering value ˋbuffer_sizeˋ.
            Possible values are:
                '':	ˋvalˋ is given unitless as a scale factor
                'px':  ˋvalˋ is given as number of pixels.
                'sr':  ˋvalˋ is given in spatial reference units (meters/degrees).
        segment_size : float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
            If None, only the corner points are used for creating the boundary geometry.
        inplace : bool, optional
            If true, the current instance will be modified (default).
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        geospade.definition.RasterGeometry
            Resized raster geometry.
        """

        if isinstance(buffer_size, (float, int)):
            buffer_size = [buffer_size]*4

        if unit not in ['', 'px', 'sr']:
            err_msg = "Unit '{}' is unknown. Please use 'px', 'sr' or ''."
            raise Exception(err_msg.format(unit))

        # first, convert the geometry to a shapely geometry
        boundary = self.to_shapely_geom()
        # then, rotate the geometry to be axis parallel if it is not axis parallel
        boundary = affinity.rotate(boundary.convex_hull, self.ori*180./np.pi, 'center')
        # loop over all edges
        scale_factors = []
        for i, buffer_size_i in enumerate(buffer_size):
            if (i == 0) or (i == 2):  # left and right edge
                if unit == '':
                    # -1 because 1 is the original size and we need negative values for shrinking a geometry
                    # resize extent (0.5 because buffer size always refers to half the edge length)
                    scale_factor = (buffer_size_i - 1) * 0.5
                elif unit == 'px':
                    scale_factor = buffer_size_i/float(self.n_cols)
                else:
                    scale_factor = buffer_size_i/float(self.width)
            else:
                if unit == '':
                    scale_factor = (buffer_size_i - 1) * 0.5
                elif unit == 'px':
                    scale_factor = buffer_size_i/float(self.n_rows)
                else:
                    scale_factor = buffer_size_i/float(self.height)

            scale_factors.append(scale_factor)

        # get extent
        vertices = list(boundary.exterior.coords)
        x_coords, y_coords = zip(*vertices)
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)

        res_min_x = round(min_x - self.width * scale_factors[0], DECIMALS)
        res_min_y = round(min_y - self.height * scale_factors[3], DECIMALS)
        res_max_x = round(max_x + self.width * scale_factors[2], DECIMALS)
        res_max_y = round(max_y + self.height * scale_factors[1], DECIMALS)

        # create new boundary geometry (counter-clockwise) and rotate it back
        new_boundary_geom = Polygon(((res_min_x, res_min_y),
                                    (res_max_x, res_min_y),
                                    (res_max_x, res_max_y),
                                    (res_min_x, res_max_y),
                                    (res_min_x, res_min_y)))
        new_boundary_geom = affinity.rotate(new_boundary_geom, -self.ori*180./np.pi, 'center')

        if segment_size is None:
            segment_size = self._segment_size

        res_raster_geom = RasterGeometry.from_geometry(new_boundary_geom, self.x_pixel_size, self.y_pixel_size, self.sref,
                                                       segment_size=segment_size, description=self.description,
                                                       geom_id=self.id, parent=self, px_origin=self.px_origin)

        if inplace:
            self._segment_size = segment_size
            self.boundary = res_raster_geom.boundary
            self.n_rows = res_raster_geom.n_rows
            self.n_cols = res_raster_geom.n_cols
            self.geotrans = res_raster_geom.geotrans
            return self
        else:
            return res_raster_geom

    def crop_px_extent(self, min_row, min_col, max_row, max_col):
        """
        Crops a given pixel extent (min_row, min_col, max_row, max_col) to the pixel limits of the raster geometry.

        Parameters
        ----------
        min_row : int
            Minimum row number.
        min_col : int
            Minimum column number.
        max_row : int
            Maximum row number.
        max_col : int
            Maximum column number.

        Returns
        -------
        min_row, min_col, max_row, max_col : int, int, int, int
            Cropped pixel extent.
        """
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.n_rows-1, max_row)
        max_col = min(self.n_cols-1, max_col)
        if min_row > max_row:
            err_msg = "Row bounds [{};{}] exceed range of possible row indexes {}."
            raise ValueError(err_msg.format(min_row, max_row, self.n_rows-1))

        if min_col > max_col:
            err_msg = "Column bounds [{};{}] exceed range of possible column indexes {}."
            raise ValueError(err_msg.format(min_col, max_col, self.n_cols-1))

        return min_row, min_col, max_row, max_col

    @_any_geom2ogr_geom
    def __contains__(self, geom):
        """
        Checks whether a given geometry is contained in the raster geometry.

        Parameters
        ----------
        geom : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Geometry to check if being within the raster geometry.

        Returns
        -------
        bool
            True if the given geometry is within the raster geometry, otherwise false.
        """

        geom_sref = geom.GetSpatialReference()
        if not self.sref.osr_sref.IsSame(geom_sref):
            err_msg = "The spatial reference systems are not equal."
            raise Exception(err_msg)

        return self.within(geom)

    def __eq__(self, other):
        """
        Checks if this and another raster geometry are equal.
        Equality holds true if the vertices, rows and columns are the same.

        Parameters
        ----------
        other: geospade.definition.RasterGeometry
            Raster geometry object to compare with.

        Returns
        -------
        bool
            True if both raster geometries are the same, otherwise false.
        """

        return self.vertices == other.vertices and \
               self.n_rows == other.n_rows and \
               self.n_cols == other.n_cols

    def __ne__(self, other):
        """
        Checks if this and another raster geometry are not equal.
        Non-equality holds true if the vertices, rows or columns differ.

        Parameters
        ----------
        other: geospade.definition.RasterGeometry
            Raster geometry object to compare with.

        Returns
        -------
        bool
            True if both raster geometries are the not the same, otherwise false.
        """

        return not self == other

    def __and__(self, other):
        """
        AND operation intersects both raster geometries.

        Parameters
        ----------
        other: geospade.definition.RasterGeometry
            Raster geometry object to intersect with.

        Returns
        -------
        geospade.definition.RasterGeometry
            Raster geometry instance defined by the bounding box of the intersection geometry.
        """

        return self.intersection_by_geom(other, inplace=False)

    def __repr__(self):
        """ str : String representation of a raster geometry as a Well Known Text (WKT) string. """

        return self.to_wkt()

    def __getitem__(self, item):
        """
        Handles indexing of a raster geometry, which is herein defined as a 2D spatial indexing via x and y coordinates.

        Parameters
        ----------
        item : 2-tuple
            Tuple containing coordinate slices (e.g., (10:100,20:200)) or coordinate values.

        Returns
        -------
        geospade.definition.RasterGeometry
            Raster geometry defined by the intersection.
        """

        if not isinstance(item, tuple) or (isinstance(item, tuple) and len(item) != 2):
            raise ValueError('Index must be a tuple containing the x and y coordinates.')
        else:
            if isinstance(item[0], slice):
                min_x = item[0].start
                max_x = item[0].stop
                segment_size = item[0].step
            else:
                min_x = item[0]
                max_x = item[0]
                segment_size = None

            if isinstance(item[1], slice):
                min_y = item[1].start
                max_y = item[1].stop
                segment_size = item[1].step
            else:
                min_y = item[1]
                max_y = item[1]

            extent = [(min_x, min_y), (max_x, max_y)]
            boundary = bbox_to_polygon(extent, osr_sref=self.sref.osr_sref, segment=segment_size)

            return self.intersection_by_geom(boundary, segment_size=segment_size, inplace=False)

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy method of `RasterGeometry` class.

        Parameters
        ----------
        memodict : dict, optional

        Returns
        -------
        RasterGeometry
        """

        n_rows = self.n_rows
        n_cols = self.n_cols
        sref = copy.deepcopy(self.sref)
        geotrans = copy.deepcopy(self.geotrans)
        geom_id = self.id
        description = self.description
        segment_size = self._segment_size
        px_origin = self.px_origin
        parent = self.parent

        return RasterGeometry(n_rows, n_cols, sref, geotrans=geotrans, geom_id=geom_id, description=description,
                              segment_size=segment_size, px_origin=px_origin, parent=parent)

# TODO: should consistency be checked, maybe optional, property is valid?
# TODO: what is exactly allowed for a grid (orientation, size of tiles, ...)?
# TODO: do I have to set raster_geoms as an optional parameter if the child class doesn't use it?
class RasterGrid(metaclass=abc.ABCMeta):
    """ Represents a homogeneous collection of `RasterGeometry` objects. """

    def __init__(self, raster_geoms, parent=None, segment_size=None):
        """
        Constructor of `RasterGrid` class.

        Parameters
        ----------
        raster_geoms : list or dict or geopandas.GeoDataFrame
            Object containing raster geometries and their IDs.
        parent : RasterGrid, optional
            Parent raster grid object.
        segment_size : float, optional
            For precision: distance in input units of longest segment of the boundary polygon.
            If None, only the corner points are used for creating the boundary geometry.
        """

        self.inventory = self.__create_inventory(raster_geoms)
        self.parent = parent
        self.geom = RasterGeometry.get_common_geometry(self.inventory.values, segment_size=segment_size)

        # get spatial reference info from first raster geometry
        self.sref = self.inventory['tile'][0].sref
        self.ori = self.inventory['tile'][0].ori

        boundary_ogr = ogr.CreateGeometryFromWkt(cascaded_union(self.inventory['geometry']).wkt)
        boundary_ogr.AssignSpatialReference(self.sref.osr_sref)


    @property
    def tile_ids(self):
        """  list : IDs of all raster geometries/tiles contained in the raster grid. """
        return sorted(list(self.inventory.index))

    @property
    def area(self):
        """ float : Computes area covered the raster grid. """
        return sum([tile.area for tile in self.inventory['tile']])

    # TODO: should raster geom be a class variable?
    @property
    def outer_extent(self):
        """ 4-tuple: Outer extent of the raster geometry containing all tiles and pixels (min_x, min_y, max_x, max_y). """

        raster_geom = RasterGeometry.get_common_geometry(list(self.inventory['tile']))

        return raster_geom.outer_extent

    def __create_inventory(self, raster_geoms):
        """
        Creates GeoPandas data frame from the given raster geometries.

        Parameters
        ----------
        raster_geoms : list or dict or geopandas.GeoDataFrame
            Object containing raster geometries and their IDs.

        Returns
        -------
        geopandas.DataFrame
            Data frame with the tile/raster geometry ID's as an index. It contains two columns, a 'geometry' column
            storing the boundary of a raster geometry and a 'tile' columns storing a `RasterGeometry` object.
        """

        if isinstance(raster_geoms, dict):
            tile_ids = list(raster_geoms.keys())
            raster_geoms = list(raster_geoms.values())
        elif isinstance(raster_geoms, list):
            tile_ids = [raster_geom.id for raster_geom in raster_geoms]
        elif isinstance(raster_geoms, geopd.GeoDataFrame):
            pass
        else:
            err_msg = "Raster geometries must be given as a list, dictionary or Pandas Series, not as '{}'"
            raise ValueError(err_msg.format(type(raster_geoms)))

        if not isinstance(raster_geoms, geopd.GeoDataFrame):
            boundaries = [shapely.wkt.loads(raster_geom.to_wkt()) for raster_geom in raster_geoms]
            inventory = geopd.GeoDataFrame({'tile': raster_geoms, 'geometry': boundaries},
                                           index=tile_ids, crs=raster_geoms[0].sref.proj4)
        else:
            inventory = raster_geoms

        return inventory

    def tile_from_id(self, tile_id):
        """ RasterGeometry : Returns raster geometry according to the given tile ID. """
        return self.inventory.loc[tile_id]['tile']

    def neighbours_from_id(self, tile_id):
        """
        list of RasterGeometry : Collects the neighbouring raster geometries/tiles
        according to the given tile ID.
        """
        idxs = self.inventory.touches(self.inventory.loc[tile_id]['geometry'])
        return list(self.inventory[idxs]['tile'])

    @_any_geom2ogr_geom
    def intersection_by_coords(self, x, y, sref=None, inplace=True, **kwargs):
        """
        Intersects a geometry with the raster grid and returns a cropped raster grid.
        Thereby, each raster geometry gets cropped to.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other` (is used by decorator).
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGrid` instance will be created.
        **kwargs
            Keyword arguments for `RasterGeometry` intersection, i.e. `segment_size` or `snap_to_grid`.

        Returns
        -------
        RasterGrid
            Cropped raster grid with cropped tiles/raster geometries.
        """

        poi = ogr.Geometry(ogr.wkbPoint)
        poi.AddPoint(x, y)

        return self.intersection_by_geom(poi, sref=sref, inplace=inplace, **kwargs)

    @_any_geom2ogr_geom
    def intersection_by_pixels(self, row, col, n_rows=1, n_cols=1, inplace=False, **kwargs):
        """
        Intersects a geometry with the raster grid and returns a cropped raster grid.
        Thereby, each raster geometry gets cropped to.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other` (is used by decorator).
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGrid` instance will be created.
        **kwargs
            Keyword arguments for `RasterGeometry` intersection, i.e. `segment_size` or `snap_to_grid`.

        Returns
        -------
        RasterGrid
            Cropped raster grid with cropped tiles/raster geometries.
        """

        px_extent = (row, col, row + n_rows, col + n_cols)
        min_row, min_col, max_row, max_col = self.geom.crop_px_extent(*px_extent)
        ul_coords = self.geom.rc2xy(min_row, min_col, px_origin="ul")
        ur_coords = self.geom.rc2xy(min_row, max_col, px_origin="ur")
        lr_coords = self.geom.rc2xy(max_row, max_col, px_origin="lr")
        ll_coords = self.geom.rc2xy(max_row, min_col, px_origin="ll")
        coords = [ul_coords, ur_coords, lr_coords, ll_coords, ul_coords]

        return self.intersection_by_geom(coords, sref=self.sref, inplace=inplace, **kwargs)

    @_any_geom2ogr_geom
    def intersection_by_geom(self, other, sref=None, inplace=False, **kwargs):
        """
        Intersects a geometry with the raster grid and returns a cropped raster grid.
        Thereby, each raster geometry gets cropped to.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from `other` (is used by decorator).
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGrid` instance will be created.
        **kwargs
            Keyword arguments for `RasterGeometry` intersection, i.e. `segment_size` or `snap_to_grid`.

        Returns
        -------
        RasterGrid
            Cropped raster grid with cropped tiles/raster geometries.
        """

        if not self.geom.intersects(other):
            wrn_msg = "Geometry does not intersect with raster grid."
            warnings.warn(wrn_msg)
            return None

        idxs = self.inventory.intersects(shapely.wkt.loads(other.ExportToWkt()))
        inventory = copy.deepcopy(self.inventory[idxs])
        intersection = lambda x: x.intersection_by_geom(other, inplace=False, sref=sref, **kwargs)
        boundary = lambda x: shapely.wkt.loads(x.boundary.ExportToWkt())
        inventory['tile'] = inventory['tile'].apply(intersection)
        inventory['geometry'] = inventory['tile'].apply(boundary)

        if inplace:
            geom = self.geom.intersection_by_geom(other, sref=sref, **kwargs)
            self.parent = self
            self.inventory = inventory
            self.geom = geom
            return self
        else:
            raster_grid = RasterGrid(inventory, parent=self)
            return raster_grid

    def plot(self, label_tiles=False, **kwargs):
        """
        Plots the raster grid on a map, i.e. all of its tiles/raster geometries.

        Parameters
        ----------

        label_tiles : bool, optional
            If true, the tile ID is plotted at the center of the raster geometry (default is False).
        **kwargs
            Keyword arguments for `RasterGeometry` intersection, i.e. `ax`, `color`, `alpha`, `proj` or `show`.
        """

        for i in range(len(self.inventory)):
            self.inventory['tile'][i].plot(label_geom=label_tiles, **kwargs)

    def __len__(self):
        """ int : Returns number tiles in the raster grid. """
        return len(self.inventory)

    def __repr__(self):
        """ str : String representation of a raster grid as a GeoPandas data frame. """
        return str(self.inventory)

    #TODO: should indexing by tile id with more than one tile id (i.e. as a list of tile ids) be supported?
    def __getitem__(self, item):
        """
        Handles indexing of a raster grid, which is herein defined as 2D spatial indexing via x and y coordinates
        or via a tile ID.

        Parameters
        ----------
        item : 2-tuple or str or int
            Tile ID (e.g., "E048N015T1") or tuple containing coordinate slices (e.g., (10:100,20:200))
            or coordinate values.

        Returns
        -------
        geospade.definition.RasterGrid
            Raster grid defined by the intersection.
        """

        if isinstance(item, str) or isinstance(item, int):
            return self.tile_from_id(item)
        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError('Index must be a tuple containing the x and y coordinates.')
            else:
                if isinstance(item[0], slice):
                    min_x = item[0].start
                    max_x = item[0].stop
                    segment_size = item[0].step
                else:
                    min_x = item[0]
                    max_x = item[0]
                    segment_size = None

                if isinstance(item[1], slice):
                    min_y = item[1].start
                    max_y = item[1].stop
                    segment_size = item[1].step
                else:
                    min_y = item[1]
                    max_y = item[1]

                extent = [(min_x, min_y), (max_x, max_y)]

                return self.intersection_by_geom(extent, segment_size=segment_size, inplace=False)
        else:
            err_msg = "Key is only allowed to be of type str or tuple."
            KeyError(err_msg)
