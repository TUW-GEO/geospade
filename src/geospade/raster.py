""" Module containing class definitions for rasters. """

import os
import sys
import ogr
import copy
import cartopy
import pandas as pd
import numpy as np
import shapely
import json
from collections import OrderedDict
import shapely.wkt
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from matplotlib.patches import Polygon as PolygonPatch

from geospade.tools import polar_point
from geospade.tools import is_rectangular
from geospade.tools import bbox_to_polygon
from geospade.tools import rasterise_polygon
from geospade.transform import build_geotransform
from geospade.transform import xy2ij
from geospade.transform import ij2xy
from geospade.transform import transform_coords
from geospade.transform import transform_geom
from geospade.crs import SpatialRef
from geospade import DECIMALS


def _align_geom(align=False):
    """
    A decorator which checks if a spatial reference is available for an `OGR.geometry` object and optionally reprojects
    the given geometry to the spatial reference of the raster geometry.


    Parameters
    ----------
    align : bool, optional
        If `align` is true, then the given geometry will be reprojected to the spatial reference of
        the raster geometry.

    Returns
    -------
    decorator
        Wrapper around `f`.

    Notes
    -----
    The OGR geometry is assumed to be given in first place

    """

    def decorator(f, *args, **kwargs):
        """
        Decorator calling wrapper function.

        Parameters
        ----------
        f : callable
            Function to wrap around/execute.

        Returns
        -------
        wrapper
            Wrapper function.

        """
        def wrapper(self, *args, **kwargs):
            geom = args[0]
            geom = geom.boundary if isinstance(geom, RasterGeometry) else geom
            other_osr_sref = geom.GetSpatialReference()  # ogr geometry is assumed to be the first argument
            if other_osr_sref is None:
                err_msg = "Spatial reference of the given geometry is not set."
                raise AttributeError(err_msg)

            # warp the geometry to the spatial reference of the raster geometry if they are not the same
            if align and hasattr(self, "sref") and not self.sref.osr_sref.IsSame(other_osr_sref):
                wrpd_geom = transform_geom(geom, self.sref)
            else:
                wrpd_geom = geom

            return f(self, wrpd_geom, *args[1:], **kwargs)

        return wrapper
    return decorator


class RasterGeometry:
    """
    Represents the geometry of a georeferenced raster.
    It describes the extent and the grid of the raster along with its spatial reference.
    The (boundary) geometry can also be used as an OGR geometry, to interact with other geometries.

    """
    def __init__(self, n_rows, n_cols, sref,
                 geotrans=(0, 1, 0, 0, 0, 1),
                 name=None,
                 description="",
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
        sref : geospade.crs.SpatialRef
            Instance representing the spatial reference of the geometry.
        geotrans : 6-tuple, optional
            GDAL geotransformation tuple.
        name : str, optional
            Name of the geometry.
        description : string, optional
            Verbal description of the geometry (defaults to "").
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        parent : RasterGeometry
            Parent `RasterGeometry` instance.

        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.geotrans = geotrans
        self.sref = sref
        self.name = name
        self.description = description
        self.parent = parent
        self.px_origin = px_origin

        # compute boundary geometry
        boundary = Polygon(self.outer_boundary_corners)
        # doing a double WKT conversion to prevent precision issues nearby machine epsilon
        boundary_ogr = ogr.CreateGeometryFromWkt(ogr.CreateGeometryFromWkt(boundary.wkt).ExportToWkt())
        boundary_ogr.AssignSpatialReference(self.sref.osr_sref)
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
            Coordinates defining the extent from the lower-left to the upper-right corner.
            This extent and the `outer_boundary_extent` definition are equal.
            (lower left x, lower left y, upper right x, upper right y)
        sref : geospade.crs.SpatialRef
            Spatial reference of the geometry/extent.
        x_pixel_size : float
            Absolute pixel size in X direction.
        y_pixel_size : float
            Absolute pixel size in Y direction.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `name`, `description`,
            or `parent`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry object defined by the given extent and pixel sizes.

        Notes
        -----
        The upper-left corner of the extent is assumed to be the (pixel) origin.

        """

        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        # calculate upper-left corner for geotransform
        ul_x, ul_y = polar_point(ll_x, ll_y, height, np.pi / 2., deg=False)
        geotrans = (ul_x, x_pixel_size, 0, ul_y, 0, -y_pixel_size)

        n_rows = int(round(height / y_pixel_size, DECIMALS))
        n_cols = int(round(width / x_pixel_size, DECIMALS))
        return cls(n_rows, n_cols, sref, geotrans=geotrans, **kwargs)

    @classmethod
    @_align_geom(align=True)
    def from_geometry(cls, geom, x_pixel_size, y_pixel_size, **kwargs):
        """
        Creates a `RasterGeometry` object from an existing geometry object.
        Since `RasterGeometry` can represent rectangles only, non-rectangular
        shapely objects get converted into their bounding boxes. Since, e.g. a `Shapely`
        geometry is not geo-referenced, the spatial reference has to be
        specified. Moreover, the resolution in both pixel grid directions has to be given.

        Parameters
        ----------
        geom : ogr.Geometry
            Geometry object from which the `RasterGeometry` object should be created.
        x_pixel_size : float
            Absolute pixel size in X direction.
        y_pixel_size : float
            Absolute pixel size in Y direction.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `name`, `description`,
            or `parent`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry object defined by the extent of the given geometry and the pixel sizes.

        Notes
        -----
        The upper-left corner of the geometry/extent is assumed to be the (pixel) origin.

        """

        sref = SpatialRef.from_osr(geom.GetSpatialReference())

        geom_ch = geom.ConvexHull()
        geom_sh = shapely.wkt.loads(geom_ch.ExportToWkt())

        if is_rectangular(geom_ch):  # the polygon can be described directly as a RasterGeometry
            geom_pts = list(geom_sh.exterior.coords)  # get boundary coordinates
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

            # based on the upper left corner, find the other corner coordinates
            if geom_sh.exterior.is_ccw:
                ul_x = xs[ul_idx]
                ul_y = ys[ul_idx]
                ur_x = xs[ul_idx-1]
                ur_y = ys[ul_idx-1]
                lr_x = xs[ul_idx-2]
                lr_y = ys[ul_idx-2]
                ll_x = xs[ul_idx-3]
                ll_y = ys[ul_idx-3]
            else:
                ul_x = xs[ul_idx]
                ul_y = ys[ul_idx]
                ll_x = xs[ul_idx - 1]
                ll_y = ys[ul_idx - 1]
                lr_x = xs[ul_idx - 2]
                lr_y = ys[ul_idx - 2]
                ur_x = xs[ul_idx - 3]
                ur_y = ys[ul_idx - 3]

            # azimuth of the bottom base of the rectangle = orientation
            rot = np.arctan2(lr_y - ll_y, lr_x - ll_x)
            rot = rot*-1 if rot >= 0. else rot

            # create GDAL geotransform
            geotrans = build_geotransform(ul_x, ul_y, x_pixel_size, -y_pixel_size, rot, deg=False)

            # define raster properties
            width = np.hypot(lr_x - ll_x, lr_y - ll_y)
            height = np.hypot(ur_x - lr_x, ur_y - lr_y)
            n_rows = int(round(height / y_pixel_size, DECIMALS))
            n_cols = int(round(width / x_pixel_size, DECIMALS))

            return RasterGeometry(n_rows, n_cols, sref, geotrans=geotrans, **kwargs)
        else:  # geom is not a rectangle
            bbox = geom_sh.bounds
            return cls.from_extent(bbox, sref, x_pixel_size, y_pixel_size, **kwargs)

    @classmethod
    def from_raster_geometries(cls, raster_geoms, **kwargs):
        """
        Creates a raster geometry, which contains all the given raster geometries given by ˋraster_geomsˋ.

        Parameters
        ----------
        raster_geoms : list of geospade.raster.RasterGeometry
            List of `RasterGeometry` objects.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry containing all given raster geometries.

        Notes
        -----
        All raster geometries must have the same spatial reference system and pixel sizes.

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

            min_x, min_y, max_x, max_y = raster_geom.outer_boundary_extent
            x_coords.extend([min_x, max_x])
            y_coords.extend([min_y, max_y])

        extent = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        return cls.from_extent(extent, sref, x_pixel_size, y_pixel_size, **kwargs)

    @classmethod
    def from_definition(cls, definition):
        """
        Creates a raster geometry from a human-readable raster geometry definition, which is a
        dictionary containing the following elements:
            - 'number_of_rows'
            - 'number_of_columns'
            - 'spatial_reference'
            - 'geotransformation'
            - 'name'
            - 'description'
            - 'pixel_origin'

        The expected values can be taken from the `RasterGeometry` constructor docs.

        Parameters
        ----------
        definition : dict
            Raster geometry definition.

        Returns
        -------
        geospade.raster.RasterGeometry

        """

        n_rows = definition['number_of_rows']
        n_cols = definition['number_of_columns']
        sref = SpatialRef(definition['spatial_reference'])
        geotrans = tuple(definition['geotransformation'])
        name = definition['name']
        description = definition['description']
        px_origin = definition['pixel_origin']

        return cls(n_rows, n_cols, sref, geotrans, name, description, px_origin)

    @classmethod
    def from_json(cls, filepath):
        """
        Creates a raster geometry from a human-readable raster geometry definition in a JSON file.
        The structure of the dictionary is described in more detail in the `from_definition` classmethod.

        Parameters
        ----------
        filepath : str
            Full JSON file path.

        Returns
        -------
        geospade.raster.RasterGeometry

        """
        with open(filepath, 'r') as tmp_file:
            json_dict = json.load(tmp_file)

        return cls.from_definition(json_dict)

    @property
    def parent_root(self):
        """ geospade.raster.RasterGeometry : Finds and returns the root/original parent `RasterGeometry`. """
        raster_geom = self
        while raster_geom.parent is not None:
            raster_geom = raster_geom.parent
        return raster_geom

    @property
    def ori(self):
        """
        float : Counter-clockwise orientation of the raster geometry in radians with respect to the
        W-E direction/horizontal.

        """
        return -np.arctan2(self.geotrans[2], self.geotrans[1])

    @property
    def is_axis_parallel(self):
        """ bool : True if the `RasterGeometry` is not rotated , i.e. it is axis-parallel. """
        return self.ori == 0.

    @property
    def ll_x(self):
        """ float : X coordinate of the lower left corner. """
        x, _ = self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin)
        return x

    @property
    def ll_y(self):
        """ float: Y coordinate of the lower left corner. """
        _, y = self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin)
        return y

    @property
    def ul_x(self):
        """ float : X coordinate of the upper left corner. """
        x, _ = self.rc2xy(0, 0, px_origin=self.px_origin)
        return x

    @property
    def ul_y(self):
        """ float: Y coordinate of the upper left corner. """
        _, y = self.rc2xy(0, 0, px_origin=self.px_origin)
        return y

    @property
    def ur_x(self):
        """ float : X coordinate of the upper right corner. """
        x, _ = self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin)
        return x

    @property
    def ur_y(self):
        """ float : Y coordinate of the upper right corner. """
        _, y = self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin)
        return y

    @property
    def lr_x(self):
        """ float : X coordinate of the upper right corner. """
        x, _ = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin)
        return x

    @property
    def lr_y(self):
        """ float : Y coordinate of the upper right corner. """
        _, y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin)
        return y

    @property
    def x_pixel_size(self):
        """ float : Pixel size in X direction. """
        return np.hypot(self.geotrans[1], self.geotrans[2])

    @property
    def y_pixel_size(self):
        """ float : Pixel size in Y direction. """
        return np.hypot(self.geotrans[4], self.geotrans[5])

    @property
    def h_pixel_size(self):
        """ float : Pixel size in W-E direction (equal to `x_pixel_size` if the `RasterGeometry` is axis-parallel). """
        return self.x_pixel_size / np.cos(self.ori)

    @property
    def v_pixel_size(self):
        """ float : Pixel size in N-S direction (equal to `y_pixel_size` if the `RasterGeometry` is axis-parallel). """
        return self.y_pixel_size / np.cos(self.ori)

    @property
    def x_size(self):
        """ float : Width of the raster geometry in world system coordinates. """
        return self.n_cols * self.x_pixel_size

    @property
    def y_size(self):
        """ float : Height of the raster geometry in world system coordinates. """
        return self.n_rows * self.y_pixel_size

    @property
    def width(self):
        """ int : Width of the raster geometry in pixels. """
        return self.n_cols

    @property
    def height(self):
        """ int : Height of the raster geometry in pixels. """
        return self.n_rows

    @property
    def shape(self):
        """ tuple : Returns the shape of the raster geometry, which is defined by the height and width in pixels. """
        return self.height, self.width

    @property
    def coord_extent(self):
        """
        4-tuple: Extent of the raster geometry with the pixel origins defined during initialisation
        (min_x, min_y, max_x, max_y).

        """
        return min([self.ll_x, self.ul_x]), min([self.ll_y, self.lr_y]), \
               max([self.ur_x, self.lr_x]), max([self.ur_y, self.ul_y])

    @property
    def outer_boundary_extent(self):
        """
        4-tuple: Outer extent of the raster geometry containing every pixel
        (min_x, min_y, max_x, max_y).

        """
        ll_x, ll_y = self.rc2xy(self.n_rows - 1, 0, px_origin="ll")
        ur_x, ur_y = self.rc2xy(0, self.n_cols - 1, px_origin="ur")
        lr_x, lr_y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr")
        ul_x, ul_y = self.rc2xy(0, 0, px_origin="ul")
        return min([ll_x, ul_x]), min([ll_y, lr_y]), max([ur_x, lr_x]), max([ur_y, ul_y])

    @property
    def size(self):
        """ int : Number of pixels covered by the raster geometry. """
        return self.width * self.height

    @property
    def centre(self):
        """ 2-tuple : Centre defined by the mass centre of the vertices. """
        return shapely.wkt.loads(self.boundary.Centroid().ExportToWkt()).coords[0]

    @property
    def outer_boundary_corners(self):
        """
        4-list of 2-tuples : A tuple containing all corners (convex hull, pixel extent) in a clock-wise order
        (lower left, lower right, upper right, upper left).

        """
        ll_x, ll_y = self.rc2xy(self.n_rows - 1, 0, px_origin="ll")
        ur_x, ur_y = self.rc2xy(0, self.n_cols - 1, px_origin="ur")
        lr_x, lr_y = self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr")
        ul_x, ul_y = self.rc2xy(0, 0, px_origin="ul")
        corner_pts = [(ll_x, ll_y),
                      (ul_x, ul_y),
                      (ur_x, ur_y),
                      (lr_x, lr_y)]
        return corner_pts

    @property
    def coord_corners(self):
        """
        4-list of 2-tuples : A tuple containing all corners (convex hull, coordinate extent) in a clock-wise order
        (lower left, lower right, upper right, upper left).

        """
        corner_pts = [(self.ll_x, self.ll_y),
                      (self.ul_x, self.ul_y),
                      (self.ur_x, self.ur_y),
                      (self.lr_x, self.lr_y)]
        return corner_pts

    @property
    def x_coords(self):
        """ np.ndarray : Returns all coordinates in X direction. """
        if self.is_axis_parallel:
            min_x, _ = self.rc2xy(0, 0)
            max_x, _ = self.rc2xy(0, self.n_cols)
            return np.arange(min_x, max_x, self.x_pixel_size)
        else:
            cols = np.array(range(self.n_cols))
            return self.rc2xy(0, cols)[0]

    @property
    def y_coords(self):
        """ np.ndarray : Returns all coordinates in Y direction. """
        if self.is_axis_parallel:
            _, min_y = self.rc2xy(self.n_rows, 0)
            _, max_y = self.rc2xy(0, 0)
            return np.arange(max_y, min_y, -self.y_pixel_size)
        else:
            rows = np.array(range(self.n_rows))
            return self.rc2xy(rows, 0)[1]

    @property
    def xy_coords(self):
        """ np.ndarray, np.ndarray : Returns meshgrid of both coordinates X and Y. """
        if self.is_axis_parallel:
            x_coords, y_coords = np.meshgrid(self.x_coords, self.y_coords, indexing='ij')
        else:
            rows, cols = np.meshgrid(np.arange(self.n_rows), np.arange(self.n_cols), indexing='ij')
            x_coords, y_coords = self.rc2xy(rows, cols)

        return x_coords, y_coords

    @property
    def boundary_ogr(self):
        """ ogr.Geometry : Returns OGR geometry representation of the boundary of a `RasterGeometry`. """
        return self.boundary

    @property
    def boundary_wkt(self):
        """ str : Returns Well Known Text (WKT) representation of the boundary of a `RasterGeometry`. """
        return self.boundary.ExportToWkt()

    @property
    def boundary_shapely(self):
        """ shapely.geometry.Polygon : Boundary of the raster geometry represented as a Shapely polygon. """
        return shapely.wkt.loads(self.boundary.ExportToWkt())

    def is_raster_coord(self, x, y, sref=None):
        """
        Checks if a point in the world system exactly lies on the raster grid spanned by the raster geometry.

        Parameters
        ----------
        x : float
            World system coordinate in X direction.
        y : float
            World system coordinate in Y direction.
        sref : SpatialRef, optional
            Spatial reference of the coordinates. Has to be given if the spatial
            reference is different than the spatial reference of the raster geometry.

        Returns
        -------
        bool :
            True if the specified X coordinate is located on the grid.
        bool :
            True if the specified Y coordinate is located on the grid.

        """

        if sref is not None:
            x, y = transform_coords(x, y, sref, self.sref.osr_sref)

        # own implementation of modulo operator due to rounding issues
        x_ratio = np.around(x / self.x_pixel_size, decimals=DECIMALS)
        y_ratio = np.around(y / self.y_pixel_size, decimals=DECIMALS)
        x_is_on_grid = abs(int(x_ratio) - x_ratio) <= (10**-DECIMALS)
        y_is_on_grid = abs(int(y_ratio) - y_ratio) <= (10**-DECIMALS)

        return x_is_on_grid, y_is_on_grid

    @_align_geom(align=True)
    def intersects(self, other):
        """
        Evaluates if this `RasterGeometry` instance and another geometry intersect.

        Parameters
        ----------
        other : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to evaluate an intersection with.

        Returns
        -------
        bool
            True if both geometries intersect, false if not.

        """
        return self.boundary.Intersects(other)

    @_align_geom(align=True)
    def touches(self, other):
        """
        Evaluates if this `RasterGeometry` instance and another geometry touch each other.

        Parameters
        ----------
        other : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to evaluate a touch operation with.

        Returns
        -------
        bool
            True if both geometries touch each other, false if not.

        """
        return self.boundary.Touches(other)

    @_align_geom(align=True)
    def within(self, other):
        """
        Evaluates if the raster geometry is fully within another geometry.

        Parameters
        ----------
        other : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry is within the raster geometry, false if not.

        """
        return self.boundary.Within(other)

    @_align_geom(align=True)
    def overlaps(self, other):
        """
        Evaluates if a geometry overlaps with the raster geometry.

        Parameters
        ----------
        other : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to evaluate an overlaps operation with.

        Returns
        -------
        bool
            True if the given geometry overlaps the raster geometry, false if not.

        """
        return self.boundary.Overlaps(other)

    @_align_geom(align=True)
    def slice_by_geom(self, other, snap_to_grid=True, inplace=False, **kwargs):
        """
        Computes an intersection figure of two geometries and returns its
        (grid axis-parallel rectangle) bounding box as a raster geometry.

        Parameters
        ----------
        other : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to intersect with.
        snap_to_grid : bool, optional
            If true, the computed corners of the intersection are rounded off to
            nearest pixel corner (default).
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGeometry` instance will be created (default).
        **kwargs :
            Additional keyword arguments for the `RasterGeometry` constructor,
            e.g. `name` or `description`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry instance defined by the bounding box of the intersection geometry.

        """
        intsct_raster_geom = None
        if not self.intersects(other):
             return intsct_raster_geom

        intersection = self.boundary.Intersection(other)
        bbox = np.around(intersection.GetEnvelope(), decimals=DECIMALS)
        if snap_to_grid:
            new_ll_x, new_ll_y = self.snap_to_grid(bbox[0], bbox[2], px_origin="ll")
            new_ur_x, new_ur_y = self.snap_to_grid(bbox[1], bbox[3], px_origin="ur")
            bbox = [new_ll_x, new_ur_x, new_ll_y, new_ur_y]

        bbox = [bbox[0], bbox[2], bbox[1], bbox[3]]
        intsct_raster_geom = self.from_extent(bbox, self.sref, self.x_pixel_size,
                                              self.y_pixel_size, px_origin=self.px_origin,
                                              parent=self, **kwargs)

        if inplace:
            self.name = intsct_raster_geom.name
            self.description = intsct_raster_geom.description
            self.boundary = intsct_raster_geom.boundary
            self.n_rows = intsct_raster_geom.n_rows
            self.n_cols = intsct_raster_geom.n_cols
            self.geotrans = intsct_raster_geom.geotrans
            intsct_raster_geom = self

        return intsct_raster_geom

    def slice_by_rc(self, row, col, height=1, width=1, inplace=False, **kwargs):
        """
        Intersects the raster geometry with a pixel extent.

        Parameters
        ----------
        row : int
            Top-left row number of the pixel window anchor.
        col : int
            Top-left column number of the pixel window anchor.
        height : int, optional
            Number of rows/height of the pixel window.
        width : int, optional
            Number of columns/width of the pixel window.
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGeometry` instance will be created (default).
        **kwargs :
            Additional keyword arguments for the `RasterGeometry` constructor,
            e.g. `name` or `description`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry instance defined by the pixel extent.

        """

        max_row = row + height
        max_col = col + width
        min_row, min_col, max_row, max_col = self.__align_pixel_extent(min_row=row, min_col=col,
                                                                       max_row=max_row, max_col=max_col)

        ul_x, ul_y = self.rc2xy(min_row, min_col, px_origin='ul')
        geotrans = build_geotransform(ul_x, ul_y, self.x_pixel_size, -self.y_pixel_size, 0)
        n_rows, n_cols = max_row - min_row + 1, max_col - min_col + 1  # +1 because max_row and max_col are set for Python indexing
        intsct_raster_geom = RasterGeometry(n_rows, n_cols, self.sref, geotrans, px_origin='ul',
                                            parent=self, **kwargs)

        if inplace:
            self.name = intsct_raster_geom.name
            self.description = intsct_raster_geom.description
            self.boundary = intsct_raster_geom.boundary
            self.n_rows = height
            self.n_cols = width
            self.geotrans = geotrans
            intsct_raster_geom = self

        return intsct_raster_geom

    def xy2rc(self, x, y, sref=None, px_origin=None):
        """
        Calculates an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x : float
            World system coordinate in X direction.
        y : float
            World system coordinate in Y direction.
        sref : SpatialRef, optional
            Spatial reference of the coordinates. Has to be given if the spatial
            reference is different than the spatial reference of the raster geometry.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")
            Defaults to None, using the class internal pixel origin.

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

        if sref is not None:
            x, y = transform_coords(x, y, sref, self.sref.osr_sref)
        px_origin = self.px_origin if px_origin is None else px_origin
        c, r = xy2ij(x, y, self.geotrans, origin=px_origin)
        return r, c

    def rc2xy(self, r, c, px_origin=None):
        """
        Returns the coordinates of the center or a corner (depending on ˋpx_originˋ) of a pixel specified
        by a row and column number.

        Parameters
        ----------
        r : int
            Pixel row number.
        c : int
            Pixel column number.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")
            Defaults to None, using the class internal pixel origin.

        Returns
        -------
        x : float
            World system coordinate in X direction.
        y : float
            World system coordinate in Y direction.

        """

        px_origin = self.px_origin if px_origin is None else px_origin
        return ij2xy(c, r, self.geotrans, origin=px_origin)

    def snap_to_grid(self, x, y, sref=None, px_origin="ul"):
        """
        Floors the given world system coordinates `x` and `y` to a coordinate point of the grid spanned by the
        raster geometry. The coordinate anchor specified by `px_origin` is then returned.

        Parameters
        ----------
        x : float
            World system coordinate in X direction.
        y : float
            World system coordinate in Y direction.
        sref : SpatialRef, optional
            Spatial reference of the coordinates. Has to be given if the spatial
            reference is different than the spatial reference of the raster geometry.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul", default)
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")

        Returns
        -------
        new_x : float
            Raster geometry grid coordinate in X direction related to the origin defined by `px_origin`.
        new_y : float
            Raster geometry grid coordinate in Y direction related to the origin defined by `px_origin`.

        """
        new_x, new_y = x, y
        x_on_grid, y_on_grid = self.is_raster_coord(x, y, sref=sref)
        if not (x_on_grid and y_on_grid):
            row, col = self.xy2rc(x, y, sref=sref, px_origin=self.px_origin)
            algnd_x, algnd_y = self.rc2xy(row, col, px_origin=px_origin)
            if not x_on_grid:
                new_x = algnd_x
            if not y_on_grid:
                new_y = algnd_y

        return new_x, new_y

    def plot(self, ax=None, facecolor='tab:red', edgecolor='black', edgewidth=1, alpha=1., proj=None,
             show=False, label_geom=False, add_country_borders=True, extent=None):
        """
        Plots the boundary of the raster geometry on a map.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        facecolor : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'tab:red').
        edgecolor : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'black').
        edgewidth : float, optional
            Width the of edge line (defaults to 1).
        alpha : float, optional
            Opacity (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes (default is None).
            If None, the projection of the spatial reference system of the raster geometry is taken.
        show : bool, optional
            If True, the plot result is shown (default is False).
        label_geom : bool, optional
            If True, the geometry name is plotted at the center of the raster geometry (default is False).
        add_country_borders : bool, optional
            If True, country borders are added to the plot (`cartopy.feature.BORDERS`) (default is False).
        extent : tuple or list, optional
            Coordinate/Map extent of the plot, given as [min_x, min_y, max_x, max_y]
            (default is None, meaning global extent).

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

        this_proj = self.sref.to_cartopy_proj()
        if proj is None:
            other_proj = this_proj
        else:
            other_proj = proj

        if ax is None:
            ax = plt.axes(projection=other_proj)
            ax.set_global()
            ax.gridlines()

        if add_country_borders:
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)

        patch = PolygonPatch(list(self.boundary_shapely.exterior.coords), facecolor=facecolor, alpha=alpha,
                            zorder=0, edgecolor=edgecolor, linewidth=edgewidth, transform=this_proj)
        ax.add_patch(patch)

        if extent is not None:
            ax.set_xlim([extent[0], extent[2]])
            ax.set_ylim([extent[1], extent[3]])

        if self.name is not None and label_geom:
            transform = this_proj._as_mpl_transform(ax)
            ax.annotate(str(self.name), xy=self.centre, xycoords=transform, va="center", ha="center")

        if show:
            plt.show()

        return ax

    def scale(self, scale_factor, inplace=False, **kwargs):
        """
        Scales the raster geometry as a whole or for each edge.
        The scaling factor always refers to an edge length of the raster geometry boundary.

        Parameters
        ----------
        scale_factor : number or list of numbers
            Scale factors, which have to be given in a clock-wise order, i.e. [left edge, top edge, right edge,
            bottom edge] or as one value.
        inplace : bool, optional
            If True, the current instance will be modified (default).
            If False, a new `RasterGeometry` instance will be created.
        **kwargs :
            Additional keyword arguments for the `RasterGeometry` constructor,
            e.g. `name` or `description`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Scaled raster geometry.

        """

        return self.resize(scale_factor, unit='', inplace=inplace, **kwargs)

    def resize(self, buffer_size, unit='px', inplace=False, **kwargs):
        """
        Resizes the raster geometry. The resizing values can be specified as a scale factor,
        in pixels, or in spatial reference units. A positive value extends, a
        negative value shrinks the original object (except for the scaling factor).

        Parameters
        ----------
        buffer_size : number or list of numbers
            Buffering values, which have to be given in a clock-wise order, i.e. [left edge, top edge, right edge,
            bottom edge] or as one value.
        unit : string, optional
            Unit of the buffering value ˋbuffer_sizeˋ.
            Possible values are:
                '':	ˋbuffer_sizeˋ is given unitless as a positive scale factor with respect to edge length.
                'px':  ˋbuffer_sizeˋ is given as number of pixels.
                'sr':  ˋbuffer_sizeˋ is given in spatial reference units (meters/degrees).
        inplace : bool, optional
            If True, the current instance will be modified.
            If False, a new `RasterGeometry` instance will be created (default).
        **kwargs :
            Additional keyword arguments for the `RasterGeometry` constructor,
            e.g. `name` or `description`.

        Returns
        -------
        geospade.raster.RasterGeometry
            Resized raster geometry.

        """

        if isinstance(buffer_size, (float, int)):
            buffer_size = [buffer_size]*4

        if unit not in ['', 'px', 'sr']:
            err_msg = "Unit '{}' is unknown. Please use 'px', 'sr' or ''."
            raise ValueError(err_msg.format(unit))

        if unit == '':
            if any([elem < 0 for elem in buffer_size]):
                err_msg = "Scale factors are only allowed to be positive numbers."
                raise ValueError(err_msg)

        # first, convert the geometry to a shapely geometry
        boundary = self.boundary_shapely
        # then, rotate the geometry to be axis parallel if it is not axis parallel
        boundary = affinity.rotate(boundary.convex_hull, self.ori*180./np.pi, 'center')
        # loop over all edges
        scale_factors = []
        for i, buffer_size_i in enumerate(buffer_size):
            if (i == 0) or (i == 2):  # left and right edge
                if unit == '':
                    # resize extent (0.5 because buffer size always refers to half the edge length)
                    scale_factor = (buffer_size_i - 1) * 0.5
                elif unit == 'px':
                    scale_factor = buffer_size_i/float(self.width)
                else:
                    scale_factor = buffer_size_i/float(self.x_size)
            else:
                if unit == '':
                    scale_factor = (buffer_size_i - 1) * 0.5
                elif unit == 'px':
                    scale_factor = buffer_size_i/float(self.height)
                else:
                    scale_factor = buffer_size_i/float(self.y_size)

            scale_factors.append(scale_factor)

        # get extent
        vertices = list(boundary.exterior.coords)
        x_coords, y_coords = zip(*vertices)
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)

        res_min_x = min_x - self.x_size * scale_factors[0]
        res_min_y = min_y - self.y_size * scale_factors[3]
        res_max_x = max_x + self.x_size * scale_factors[2]
        res_max_y = max_y + self.y_size * scale_factors[1]

        # create new boundary geometry (clockwise) and rotate it back
        new_boundary_geom = Polygon(((res_min_x, res_min_y),
                                    (res_min_x, res_max_y),
                                    (res_max_x, res_max_y),
                                    (res_max_x, res_min_y),
                                    (res_min_x, res_min_y)))
        new_boundary_geom = affinity.rotate(new_boundary_geom, -self.ori*180./np.pi, 'center')
        new_boundary_geom = ogr.CreateGeometryFromWkt(new_boundary_geom.wkt)
        new_boundary_geom.AssignSpatialReference(self.sref.osr_sref)

        res_raster_geom = self.from_geometry(new_boundary_geom, self.x_pixel_size, self.y_pixel_size,
                                             px_origin=self.px_origin, parent=self, **kwargs)

        if inplace:
            self.name = res_raster_geom.name
            self.description = res_raster_geom.description
            self.boundary = res_raster_geom.boundary
            self.n_rows = res_raster_geom.n_rows
            self.n_cols = res_raster_geom.n_cols
            self.geotrans = res_raster_geom.geotrans
            res_raster_geom = self

        return res_raster_geom

    def to_definition(self):
        """
        Creates a human-readable definition of the raster geometry.

        Returns
        -------
        dict

        """
        raster_geom_dict = dict()
        raster_geom_dict['name'] = self.name
        raster_geom_dict['number_of_rows'] = self.n_rows
        raster_geom_dict['number_of_columns'] = self.n_cols
        raster_geom_dict['spatial_reference'] = self.sref.to_proj4_dict()
        raster_geom_dict['geotransformation'] = self.geotrans
        raster_geom_dict['pixel_origin'] = self.px_origin
        raster_geom_dict['description'] = self.description

        return raster_geom_dict

    def to_json(self, filepath):
        """
        Creates a human-readable definition of the raster geometry in JSON format and writes it to disk.

        Parameters
        ----------
        filepath : str
            Full JSON file path.

        """
        json_dict = self.to_definition()
        with open(filepath, 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)

    def __align_pixel_extent(self, min_row=0, min_col=0, max_row=1, max_col=1):
        """
        Crops a given pixel extent (min_row, min_col, max_row, max_col) to the pixel limits of the raster geometry.

        Parameters
        ----------
        min_row : int, optional
            Minimum row number (defaults to 0).
        min_col : int, optional
            Minimum column number (defaults to 0).
        max_row : int, optional
            Maximum row number (defaults to 1).
        max_col : int, optional
            Maximum column number (defaults to 1).

        Returns
        -------
        min_row, min_col, max_row, max_col : int, int, int, int
            Pixel extent not crossing the bounds of the raster geometry.

        """
        max_row = max_row if max_row is not None else self.n_rows-1
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.n_rows-1, max_row)  # -1 because of Python indexing
        max_col = min(self.n_cols-1, max_col)  # -1 because of Python indexing
        if min_row > max_row:
            err_msg = "The minimum row number is not allowed to be larger than the maximum row number."
            raise ValueError(err_msg)

        if min_col > max_col:
            err_msg = "The minimum column number is not allowed to be larger than the maximum column number."
            raise ValueError(err_msg)

        return min_row, min_col, max_row, max_col

    @_align_geom(align=False)
    def __contains__(self, geom):
        """
        Checks whether a given geometry is contained in the raster geometry.

        Parameters
        ----------
        geom : geospade.raster.RasterGeometry or ogr.Geometry
            Geometry to check if being within the raster geometry.

        Returns
        -------
        bool
            True if the given geometry is within the raster geometry, otherwise false.

        """

        geom_sref = geom.GetSpatialReference()
        if not self.sref.osr_sref.IsSame(geom_sref):
            err_msg = "The spatial reference systems are not equal."
            raise ValueError(err_msg)

        return geom.Within(self.boundary)

    def __eq__(self, other):
        """
        Checks if this and another raster geometry are equal.
        Equality holds true if the vertices, rows and columns are the same.

        Parameters
        ----------
        other : geospade.raster.RasterGeometry
            Raster geometry to compare with.

        Returns
        -------
        bool
            True if both raster geometries are the same, otherwise false.

        """

        return self.outer_boundary_corners == other.outer_boundary_corners and \
               self.n_rows == other.n_rows and \
               self.n_cols == other.n_cols

    def __ne__(self, other):
        """
        Checks if this and another raster geometry are not equal.
        Non-equality holds true if the vertices, rows or columns differ.

        Parameters
        ----------
        other : geospade.raster.RasterGeometry
            Raster geometry object to compare with.

        Returns
        -------
        bool
            True if both raster geometries are the not the same, otherwise false.

        """

        return not self == other

    @_align_geom(align=False)
    def __and__(self, other):
        """
        AND operation intersects both raster geometries.

        Parameters
        ----------
        other : geospade.raster.RasterGeometry or ogr.Geometry
            Raster geometry object to intersect with.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry instance defined by the bounding box of the intersection geometry.

        """
        geom_sref = other.GetSpatialReference()
        if not self.sref.osr_sref.IsSame(geom_sref):
            err_msg = "The spatial reference systems are not equal."
            raise ValueError(err_msg)

        return self.slice_by_geom(other, inplace=False)

    def __str__(self):
        """ str : Representation of a raster geometry as a Well Known Text (WKT) string. """

        return self.boundary_wkt

    def __getitem__(self, item):
        """
        Handles indexing of a raster geometry, which is herein defined as a 2D spatial indexing via X and Y coordinates
        or pixel slicing.

        Parameters
        ----------
        item : 2-tuple or 3-tuple
            - 2-tuple: contains two indexes/slices for row and column pixel indexing/slicing.
            - 3-tuple: contains two coordinates/slices and one entry for a `SpatialRef` instance defining
                       the spatial reference system of the coordinates.

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry defined by the intersection.

        """

        err_msg = "The given way of indexing is not supported. Only two (pixel) or " \
                  "three (coordinates) indexes are supported."
        intsct_raster_geom = None
        if isinstance(item, tuple) and (len(item) in [2, 3]):
            if isinstance(item[0], slice):
                min_f_idx = item[0].start
                max_f_idx = item[0].stop
            else:
                min_f_idx = item[0]
                max_f_idx = item[0]

            if isinstance(item[1], slice):
                min_s_idx = item[1].start
                max_s_idx = item[1].stop
            else:
                min_s_idx = item[1]
                max_s_idx = item[1]

            if len(item) == 2:
                height = max_f_idx - min_f_idx - 1
                width = max_s_idx - min_s_idx - 1
                intsct_raster_geom = self.slice_by_rc(min_f_idx, min_s_idx, height, width, inplace=False)
            elif len(item) == 3:
                sref = item[2]
                extent = [min_f_idx, min_s_idx, max_f_idx, max_s_idx]
                intsct_raster_geom = self.from_extent(extent, sref, self.x_pixel_size, self.y_pixel_size,
                                                      parent=self)
            else:
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)

        return intsct_raster_geom

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy method of `RasterGeometry` class.

        Parameters
        ----------
        memodict : dict, optional

        Returns
        -------
        geospade.raster.RasterGeometry

        """

        n_rows = self.n_rows
        n_cols = self.n_cols
        sref = copy.deepcopy(self.sref)
        geotrans = copy.deepcopy(self.geotrans)
        name = self.name
        description = self.description
        px_origin = self.px_origin
        parent = self.parent

        return RasterGeometry(n_rows, n_cols, sref, geotrans, name, description, px_origin, parent)


class Tile(RasterGeometry):
    """ A light wrapper around a raster geometry to realise a tile - mosaic relationship. """
    def __init__(self, n_rows, n_cols, sref,
                 geotrans=(0, 1, 0, 0, 0, 1),
                 mosaic_topology="INNER",
                 active=True,
                 metadata=None,
                 name=None,
                 description="",
                 px_origin="ul",
                 parent=None):
        """
        Constructor of the `Tile` class.

        Parameters
        ----------
        n_rows : int
            Number of pixel rows.
        n_cols : int
            Number of pixel columns.
        sref : geospade.spatial_ref.SpatialRef
            Instance representing the spatial reference of the geometry.
        geotrans : 6-tuple, optional
            GDAL geotransformation tuple.
        mosaic_topology : str, optional
            String defining the relation between the mosaic boundary and the tile.
            It can be None, "INNER", "OUTER" or "BOUNDARY". Defaults to "INNER".
        active : bool, optional
            Defines if a tile is active or not (defaults to True).
        metadata : dict, optional
            Dictionary containing tile-based metadata.
        name : int or str, optional
            Name of the geometry.
        description : string, optional
            Verbal description of the geometry (defaults to "").
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        parent : geospade.raster.Tile
            Parent `Tile` instance.

        """
        super().__init__(n_rows, n_cols, sref, geotrans, name, description, px_origin, parent)
        if mosaic_topology not in ["INNER", "OUTER", "BOUNDARY", None]:
            err_msg = "Relation '{}' is not valid. Use None, 'INNER', 'OUTER' or 'BOUNDARY'."
            raise ValueError(err_msg.format(mosaic_topology))
        self.mosaic_topology = mosaic_topology
        self.active = active
        self.metadata = {} if metadata is None else metadata
        self._mask = None

    @property
    def mask(self):
        """ np.ndarray : Binary 2D array representing a pixel mask. """
        return self._mask

    @mask.setter
    def mask(self, mask):
        """
        Assigns externally generated mask to this tile.

        Parameters
        ----------
        mask : np.ndarray
            Binary 2D array.

        """
        if mask.shape != self.shape:
            err_msg = "Shape mismatch of mask {} and tile {}.".format(mask.shape, self.shape)
            raise ValueError(err_msg)
        else:
            self._mask = mask

    @classmethod
    def from_definition(cls, definition):
        """
        Creates a tile from a human-readable tile definition, which is a
        dictionary containing the following elements:
            - 'number_of_rows'
            - 'number_of_columns'
            - 'spatial_reference'
            - 'geotransformation'
            - 'mosaic_topology'
            - 'active'
            - 'metadata'
            - 'name'
            - 'description'
            - 'pixel_origin'

        The expected values can be taken from the `RasterGeometry` and `Tile` constructor docs.

        Parameters
        ----------
        definition : dict
            Tile definition.

        Returns
        -------
        geospade.raster.Tile

        """

        n_rows = definition['number_of_rows']
        n_cols = definition['number_of_columns']
        sref = SpatialRef(definition['spatial_reference'])
        geotrans = tuple(definition['geotransformation'])
        mosaic_topology = definition['mosaic_topology']
        active = definition['active'].lower() == 'true'
        metadata = definition['metadata']
        name = definition['name']
        description = definition['description']
        px_origin = definition['pixel_origin']

        return cls(n_rows, n_cols, sref, geotrans, mosaic_topology, active, metadata,
                   name, description, px_origin)

    def to_definition(self):
        """
        Creates a human-readable definition of the tile.

        Returns
        -------
        dict

        """
        tile_dict = super().to_definition()
        tile_dict['active'] = str(self.active).lower()
        tile_dict['mosaic_topology'] = self.mosaic_topology
        tile_dict['metadata'] = self.metadata

        return tile_dict


class MosaicGeometry:
    """
    Represents an irregular mosaic of tiles.
    Tiles are not allowed to intersect with each other.

    """
    __type = 'irregular'

    def __init__(self, tiles, boundary=None, adjacency_matrix=None, name='', description="",
                 check_consistency=True):
        """
        Constructor of `MosaicGeometry` class.

        Parameters
        ----------
        tiles : list of geospade.raster.Tile
            List of tiles.
        boundary : ogr.Geometry, optional
            Strictly defined boundary of the mosaic, i.e. it defines where coordinates are valid/belong to the grid and
            where not. If it is None, the cascaded union of all tiles defines the boundary.
        adjacency_matrix : np.array, optional
            Adjacency matrix given as a boolean array defining the direct neighbourhood relationships of the given tiles.
            It needs to have the same size as the given number of tiles. If it is None, an adjacency matrix is created
            on-the-fly (default).
        name : str, optional
            Name of the mosaic.
        description : string, optional
            Verbal description of the mosaic (defaults to "").
        check_consistency : bool, optional
            If True, the tiles are checked for consistency, i.e. to be non-overlapping (defaults to True).

        """

        if adjacency_matrix is not None:
            if adjacency_matrix.ndim != 2:
                err_msg = "Adjacency matrix is expected to be passed as a 2D numpy array."
                raise ValueError(err_msg)

        if check_consistency:
            is_consistent = self.__check_consistency(tiles)
            if not is_consistent:
                err_msg = "Tiles are not allowed to overlap!"
                raise ValueError(err_msg)

        ref_tile = tiles[0]
        self.sref = ref_tile.sref
        self.ori = ref_tile.ori
        self.name = name
        self.description = description

        if boundary is None:
            tile_boundaries = [tile.boundary_shapely for tile in tiles]
            boundary = ogr.CreateGeometryFromWkt(cascaded_union(tile_boundaries).wkt)
            boundary.AssignSpatialReference(self.sref.osr_sref)
        self.boundary = boundary

        self._adjacency_matrix = adjacency_matrix
        if adjacency_matrix is None:
            self._adjacency_matrix = self.__build_adjacency_matrix(tiles)

        # create dataframe as an internal tile inventory
        self._tiles = self.__build_df(tiles)

    @property
    def all_tiles(self):
        """ list : All tiles of the mosaic. """
        return list(self._tiles['tile'])

    @property
    def all_tile_names(self):
        """ list : All tile names of the mosaic. """
        return list(self._tiles.index)

    @property
    def tiles(self):
        """ list : All active tiles of the mosaic. """
        return list(self._tiles[self._tiles['active']]['tile'])

    @property
    def tile_names(self):
        """ list : All active tile names of the mosaic. """
        return list(self._tiles[self._tiles['active']].index)

    @classmethod
    def from_definition(cls, definition, tile_class=Tile, check_consistency=True):
        """
        Creates a mosaic geometry from a human-readable mosaic definition, which is a
        dictionary containing the following elements:
            - 'type'
            - 'boundary'
            - 'name'
            - 'description'
            - 'adjacency_matrix'
            - 'tiles'

        The expected values can be taken from the `MosaicGeometry`, `Tile`, and `RasterGeometry` constructor docs.

        Parameters
        ----------
        definition : dict
            Human-readable definition of a mosaic.
        tile_class : class, optional
            Class inheriting from `Tile` (default).
        check_consistency : bool, optional
            If True, the tiles are checked for consistency, i.e. to be non-overlapping (defaults to True).

        Returns
        -------
        geospade.raster.MosaicGeometry

        """
        mosaic_type = definition['type']
        if mosaic_type != cls.__type:
            err_msg = "Mosaic type of definition '{}' does not match expected mosaic type '{}'".format(mosaic_type,
                                                                                                       cls.__type)
            raise ValueError(err_msg)
        mosaic_boundary = shapely.wkt.loads(definition['boundary'])
        mosaic_name = definition['name']
        description = definition['description']
        adjacency_matrix = None if definition['adjacency_matrix'] is None else np.array(definition['adjacency_matrix'])

        tiles = []
        for key in definition['tiles'].keys():
            tiles.append(tile_class.from_definition(definition['tiles'][key]))

        return cls(tiles, mosaic_boundary, adjacency_matrix, mosaic_name, description, check_consistency)

    @classmethod
    def from_json(cls, filepath, tile_class=Tile, check_consistency=True):
        """
        Creates a mosaic geometry from disk.

        Parameters
        ----------
        filepath : str
            Full JSON file path.
        tile_class : class, optional
            Class inheriting from `Tile` (default).
        check_consistency : bool, optional
            If True, the tiles are checked for consistency, i.e. to be non-overlapping (defaults to True).

        Returns
        -------
        geospade.raster.MosaicGeometry

        """

        if not os.path.exists(filepath):
            err_msg = "'{}' does not exist.".format(filepath)
            raise FileNotFoundError(err_msg)
        with open(filepath, 'r') as def_file:
            definition = json.load(def_file)

        return cls.from_definition(definition, tile_class, check_consistency)

    def xy2tile(self, x, y, sref=None):
        """
        Returns the tile intersecting with the given world system coordinates. If the coordinates are outside the
        mosaic boundary, no tile is returned.

        Parameters
        ----------
        x : number
            World system coordinate in X direction.
        y : number
            World system coordinate in Y direction.
        sref : SpatialRef, optional
            Spatial reference system of the world system coordinates. If None, the spatial reference system of the
            coordinates and the spatial reference system of the mosaic are assumed to be the same (default).

        Returns
        -------
        geospade.raster.Tile :
            Tile intersecting/matching with the given world system coordinates.

        """
        # create OGR point
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x, y)
        osr_sref = self.sref.osr_sref if sref is None else sref.osr_sref
        point.AssignSpatialReference(osr_sref)

        tile_oi = None
        if point.Intersects(self.boundary):  # point is inside grid definition
            for tile in self.all_tiles:
                if tile.intersects(point):  # point is inside tile
                    tile_oi = self._mask_tile(tile)
                    break

        return tile_oi

    def name2tile(self, tile_name, active_only=True, apply_mask=True):
        """
        Converts a tile name to a tile.

        Parameters
        ----------
        tile_name : str
            Tile name.
        active_only : bool, optional
            If true, only active tiles are returned (default).
        apply_mask : bool, optional
            If true, tiles will have a mask for reflecting the relation with the exact mosaic boundary (default).

        Returns
        -------
        geospade.raster.Tile :
            Tile referring to `tile_name`.

        """
        if tile_name not in self._tiles.index:
            err_msg = "Tile name '{}' is not available.".format(tile_name)
            raise KeyError(err_msg)
        tile_oi = self._tiles.loc[tile_name]['tile']
        if apply_mask:
            tile_oi = self._mask_tile(tile_oi)
        if active_only and not self._tiles.loc[tile_name, 'active']:
            tile_oi = None

        return tile_oi

    def get_neighbouring_tiles(self, tile_name, active_only=True, apply_mask=True):
        """
        Returns all tiles being in the direct neighbourhood of the tile with the given tile name.

        Parameters
        ----------
        tile_name : str or int
            Tile name.
        active_only : bool, optional
            If true, only active tiles are returned (default).
        apply_mask : bool, optional
            If true, tiles will have a mask for reflecting the relation with the exact mosaic boundary (default).

        Returns
        -------
        list :
            List of tiles being in the neighbourhood of `tile_name`.

        """
        tile_idx = self.all_tile_names.index(tile_name)
        nbr_idxs = self._adjacency_matrix[tile_idx, :]
        nbr_tiles_df = self._tiles.iloc[nbr_idxs]
        if active_only:
            nbr_tiles = list(nbr_tiles_df[nbr_tiles_df['active']]['tile'])
        else:
            nbr_tiles = list(nbr_tiles_df['tile'])

        if apply_mask:
            nbr_tiles = [self._mask_tile(nbr_tile) for nbr_tile in nbr_tiles]

        nbr_tiles = dict([(nbr_tile.name, nbr_tile) for nbr_tile in nbr_tiles])
        return nbr_tiles

    @_align_geom(align=True)
    def select_tiles_by_geom(self, geom, active_only=True, apply_mask=True):
        """
        Computes an intersection figure of the mosaic and another geometry and returns the tiles intersecting with this
        figure.

        Parameters
        ----------
        geom : ogr.Geometry
            Other geometry to intersect with.
        active_only : bool, optional
            If true, only active tiles are returned (default).
        apply_mask : bool, optional
            If true, tiles will have a mask for reflecting the relation with the exact mosaic boundary (default).

        Returns
        -------
        dict
            Dictionary of tiles (key=tilename, value=tile object).

        """
        selected_tiles = dict()
        tile_ids_done = []
        tiles = self.tiles if active_only else self.all_tiles
        for tile in tiles:
            if tile.intersects(geom):
                tile_ids_done.append(tile.name)
                tile = self._mask_tile(tile) if apply_mask else tile
                selected_tiles[tile.name] = tile
                nbr_tiles = self.get_neighbouring_tiles(tile.name, active_only).values()
                while len(nbr_tiles) > 0:
                    new_nbr_tiles = []
                    for nbr_tile in nbr_tiles:
                        if nbr_tile.name not in tile_ids_done:
                            tile_ids_done.append(nbr_tile.name)
                            if nbr_tile.intersects(geom):
                                nbr_tile = self._mask_tile(nbr_tile) if apply_mask else nbr_tile
                                selected_tiles[nbr_tile.name] = nbr_tile
                                next_nbr_tiles = self.get_neighbouring_tiles(nbr_tile.name, active_only).values()
                                next_nbr_tiles = [nnbr_tile for nnbr_tile in next_nbr_tiles
                                                  if nnbr_tile.name not in tile_ids_done]
                                new_nbr_tiles.extend(next_nbr_tiles)

                    nbr_tiles = new_nbr_tiles
                break

        return selected_tiles

    @_align_geom(align=True)
    def slice_tiles_by_geom(self, geom, active_only=True):
        """
        Computes an intersection figure of the mosaic and another geometry and returns a list of intersected tiles.

        Parameters
        ----------
        geom : ogr.Geometry
            Other geometry to intersect with.
        active_only : bool, optional
            If true, only active tiles are returned (default).

        Returns
        -------
        dict
            Dictionary of intersected tiles (key=tilename, value=tile object).

        Notes
        -----
        This operation dissolves the relationship between the mosaic and a tile.

        """
        tiles = self.select_tiles_by_geom(geom, active_only)
        intsctd_tiles = dict()
        for i, tile in enumerate(tiles.values()):
            # intersected tile does not have a relation with the original mosaic tile anymore
            intsctd_tile = tile.slice_by_geom(geom, True, False, name=i,
                                              mosaic_topology=None,
                                              active=False)
            intsctd_tiles[intsctd_tile.name] = intsctd_tile

        return intsctd_tiles

    @_align_geom(align=True)
    def slice_by_geom(self, geom):
        """
        Activates all mosaic tiles intersecting with the given geometry.

        Parameters
        ----------
        geom : ogr.Geometry or geospade.raster.RasterGeometry
            Other geometry to intersect with.

        """
        tiles = self.select_tiles_by_geom(geom, False, False)
        tile_names = list(tiles.keys())
        self._tiles['active'] = False
        self._tiles.loc[tile_names, 'active'] = True

    def filter_tile_metadata(self, metadata):
        """
        Activates all mosaic tiles matching the given metadata dictionary.

        Parameters
        ----------
        metadata : dict
            Tile metadata dictionary.

        """
        tile_df = self._tiles
        for key, value in metadata.items():
            if key not in self._tiles.columns:
                err_msg = "Metadata key '{}' is not available".format(key)
                raise KeyError(err_msg)
            else:
                tile_df = tile_df[tile_df[key] == value]

        self._tiles['active'] = False
        self._tiles.loc[tile_df.index, 'active'] = True

    def plot(self, ax=None, facecolor='tab:red', edgecolor='black', edgewidth=1, alpha=1., proj=None,
             show=False, label_tiles=False, add_country_borders=True, extent=None, active_only=True,
             plot_boundary=True):
        """
        Plots the tiles of the irregular mosaic geometry on a map.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        facecolor : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'tab:red').
        edgecolor : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'black').
        edgewidth : float, optional
            Width the of edge line (defaults to 1).
        alpha : float, optional
            Opacity of the boundary polygon (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes (default is None).
            If None, the projection of the spatial reference system of the mosaic is taken.
        show : bool, optional
            If True, the plot result is shown (default is False).
        label_tiles : bool, optional
            If True, the tile names are plotted at the center of each tile (default is False).
        add_country_borders : bool, optional
            If True, country borders are added to the plot (`cartopy.feature.BORDERS`) (default is False).
        extent : tuple or list, optional
            Coordinate/Map extent of the plot, given as [min_x, min_y, max_x, max_y]
            (default is None, meaning global extent).
        active_only : bool, optional
            If true, only active tiles are plotted (default).
        plot_boundary : bool, optional
            If true, the mosaic boundary is plotted (default).

        Returns
        -------
        matplotlib.pyplot.axes
            Matplotlib axis containing a Cartopy map with the plotted irregular mosaic tiles.

        """

        if 'matplotlib' in sys.modules:
            import matplotlib.pyplot as plt
        else:
            err_msg = "Module 'matplotlib' is mandatory for plotting a MosaicGeometry object."
            raise ImportError(err_msg)

        tiles = self.tiles if active_only else self.all_tiles
        for tile in tiles:
            ax = tile.plot(ax, facecolor, edgecolor, edgewidth, alpha, proj,
                           False, label_tiles, add_country_borders, extent)

        if plot_boundary:
            mosaic_boundary = shapely.wkt.loads(self.boundary.ExportToWkt())
            x_coords_bound, y_coords_bound = list(zip(*mosaic_boundary.exterior.coords))
            this_proj = self.sref.to_cartopy_proj()
            transform = this_proj._as_mpl_transform(ax)
            ax.plot(x_coords_bound, y_coords_bound, color='k', linewidth=2, transform=transform)

        if show:
            plt.show()

        return ax

    def to_definition(self):
        """
        Creates a human-readable mosaic definition.

        Returns
        -------
        dict

        """
        definition = OrderedDict()
        definition['name'] = self.name
        definition['description'] = self.description
        definition['tiles'] = OrderedDict()
        definition['type'] = self.__type
        definition['adjacency_matrix'] = self._adjacency_matrix.tolist()
        definition['boundary'] = self.boundary.ExportToWkt()

        for i, tile in enumerate(self.all_tiles):
            tile.active = self._tiles.iloc[i]['active']
            definition['tiles'][i] = tile.to_definition()

        return definition

    def to_json(self, filepath):
        """
        Dumps mosaic represented by its human-readable definition and adjacency matrix to disk.

        Parameters
        ----------
        filepath : str
            Full JSON file path.

        """
        with open(filepath, 'w') as def_file:
            json.dump(self.to_definition(), def_file, indent=4)

    def _mask_tile(self, tile):
        """
        Creates a pixel mask for the specified tile, where 1 is inside
        and 0 outside the mosaic boundary.

        Parameters
        ----------
        tile : geospade.raster.Tile
            Tile to mask/intersect with the mosaic boundary.

        Returns
        -------
        tile : geospade.raster.Tile
            Tile with the assigned mask.

        """
        if tile.mask is None:
            if tile.mosaic_topology == 'INNER':
                tile.mask = np.ones(tile.shape)
            elif tile.mosaic_topology == 'OUTER':
                tile.mask = np.zeros(tile.shape)
            elif tile.mosaic_topology == 'BOUNDARY':
                intrsct_geom = self.boundary.Intersection(tile.boundary)
                # first, using 'outer_boundary_extent' as a pixel buffer for generating the rasterised
                # pixel skeleton
                # second, reduce this pixel buffer again to the coordinate extent by skipping the last
                # row and column
                tile.mask = rasterise_polygon(intrsct_geom, tile.x_pixel_size, tile.y_pixel_size,
                                              tile.outer_boundary_extent)[:-1, :-1]

        return tile

    def __build_df(self, tiles):
        """
        Converts a list of tiles to a data frame.

        Parameters
        ----------
        tiles : list of geospade.raster.Tile
            List of tiles.

        Returns
        -------
        pd.Dataframe
            Dataframe storing relevant tile information.

        """
        ref_tile = tiles[0]
        columns = ['tile', 'active', 'topology'] + list(ref_tile.metadata.keys())
        tile_dict = dict()
        for column in columns:
            tile_dict[column] = []

        index = []
        for tile in tiles:
            tile_dict['tile'].append(tile)
            index.append(tile.name)
            tile_dict['active'].append(tile.active)
            tile_dict['topology'].append(tile.mosaic_topology)
            for key, value in tile.metadata.items():
                if key not in columns:
                    err_msg = "Metadata key '{}' differs from existing keys.".format(key)
                    raise KeyError(err_msg)
                else:
                    tile_dict[key].append(value)

        return pd.DataFrame(tile_dict, index=index)

    def __build_adjacency_matrix(self, tiles):
        """
        np.array : 2D numpy array representing an adjacency matrix, i.e. it contains information which tiles lay
        within the direct neighbourhood of one tile.

        """
        n_tiles = len(tiles)
        adjacency_matrix = np.zeros((n_tiles, n_tiles), dtype=bool)
        for i in range(n_tiles):
            for j in range(i, n_tiles):
                if i != j:
                    if tiles[i].touches(tiles[j]):
                        adjacency_matrix[i, j] = True
                        adjacency_matrix[j, i] = True

        return adjacency_matrix

    def __check_consistency(self, tiles):
        """
        Checks if the given tiles are consistent, which is the case if they do not overlap.

        Parameters
        ----------
        tiles : list of geospade.raster.Tile
            List of tiles.

        Returns
        -------
        is_consistent : bool
            True if the mosaic is valid/consistent, False if not.

        """
        is_consistent = True
        n_tiles = len(tiles)
        for i in range(n_tiles):
            for j in range(i, n_tiles):
                if i != j:
                    if tiles[i].overlaps(tiles[j]):
                        is_consistent = False
                        break
            if not is_consistent:
                break

        return is_consistent

    def __getitem__(self, item):
        """
        Handles slicing of a mosaic, which is herein defined as either 1D indexing with a tile name or
        3D spatial indexing via X and Y coordinates and their spatial reference.

        Parameters
        ----------
        item : str or 3-tuple
            - str : a tile name
            - 3-tuple : contains two coordinates/slices and one entry for a `SpatialRef` instance defining
                        the spatial reference system of the coordinates

        Returns
        -------
        `Tile` or dict of (tile name, geospade.raster.Tile) pairs:
            Returns either one or multiple tiles from the mosaic.

        """

        ret_geom = None

        if isinstance(item, (str, int)):
            ret_geom = self.name2tile(item)
        elif isinstance(item, tuple) and (len(item) == 3):
            if isinstance(item[0], slice):
                min_f_idx = item[0].start
                max_f_idx = item[0].stop
            else:
                min_f_idx = item[0]
                max_f_idx = item[0]

            if isinstance(item[1], slice):
                min_s_idx = item[1].start
                max_s_idx = item[1].stop
            else:
                min_s_idx = item[1]
                max_s_idx = item[1]

            sref = item[2]
            extent = [(min_f_idx, min_s_idx), (max_f_idx, max_s_idx)]
            boundary = bbox_to_polygon(extent, sref)
            ret_geom = self.select_tiles_by_geom(boundary)
        else:
            err_msg = "The given way of indexing is not supported. Only one (tile name) or " \
                      "three (coordinates) indexes are supported."
            raise ValueError(err_msg)

        return ret_geom


class RegularMosaicGeometry(MosaicGeometry):
    """
    Represents a regular, homogeneous mosaic of tiles. This means tiles are not allowed to
    overlap or vary in size.

    """
    __type = 'regular'

    def __init__(self, tiles, boundary=None, adjacency_matrix=None, name=None, description="", check_consistency=True):
        """
        Constructor of `RegularMosaicGeometry` class.

        Parameters
        ----------
        tiles : list of geospade.raster.Tile
            List of tiles.
        boundary : ogr.Geometry, optional
            Strictly defined boundary of the mosaic, i.e. it defines where coordinates are valid/belong to the grid and
            where not. If it is None, the cascaded union of all tiles defines the boundary.
        adjacency_matrix : np.array, optional
            Adjacency matrix given as a boolean array defining the direct neighbourhood relationships of the given tiles.
            It needs have the same size as the given number of tiles. If it is None, an adjacency matrix is created
            on-the-fly (default).
        name : str, optional
            Name of the mosaic.
        description : string, optional
            Verbal description of the mosaic geometry (defaults to "").
        check_consistency : bool, optional
            If True, the tiles are checked for consistency, i.e. to be non-overlapping and of the same shape
            (defaults to True).

        """

        if check_consistency:
            is_consistent = self.__check_consistency(tiles)
            if not is_consistent:
                err_msg = "Tiles differ in their shape."
                raise ValueError(err_msg)

        if adjacency_matrix is None:
            adjacency_matrix = self.__build_adjacency_matrix(tiles)

        super(RegularMosaicGeometry, self).__init__(tiles, boundary, adjacency_matrix, name, description,
                                                    check_consistency)

        ul_xs, ul_ys = zip(*[(tile.ul_x, tile.ul_y) for tile in tiles])
        ul_x, ul_y = min(ul_xs), max(ul_ys)
        ref_tile = tiles[0]
        self.x_pixel_size = ref_tile.x_pixel_size
        self.y_pixel_size = ref_tile.y_pixel_size
        geotrans = ref_tile.geotrans
        x_tile_size = ref_tile.x_size
        y_tile_size = ref_tile.y_size
        raster_geom_geotrans = (ul_x, x_tile_size, geotrans[2],
                                ul_y, geotrans[4], -y_tile_size)
        n_rows, n_cols = adjacency_matrix.shape
        self._raster_geom = RasterGeometry(n_rows, n_cols, self.sref, geotrans=raster_geom_geotrans)

    @classmethod
    def from_rectangular_definition(cls, n_rows, n_cols, x_tile_size, y_tile_size, sref,
                                    geotrans=(0, 1, 0, 0, 0, 1), tile_class=Tile, tile_kwargs=None,
                                    name_frmt="S{:03d}W{:03d}", boundary=None, **kwargs):
        """
        Creates a `RegularMosaicGeometry` instance from a well-defined definition of mosaic, i.e. origin, tile sizes,
        number of tiles and spatial reference system.

        Parameters
        ----------
        n_rows : int
            Number of mosaic tiles on Y direction.
        n_cols : int
            Number of mosaic tiles on X direction.
        x_tile_size : number
            Size/width of one tile given in X direction and in world system units.
        y_tile_size : number
            Size/width of one tile given in Y direction and in world system units.
        sref : SpatialRef
            Spatial reference system of the mosaic.
        geotrans : 6-tuple, optional
            GDAL geotransformation tuple containing information about the origin of the mosaic, its pixel sampling and
            its orientation (defaults to (0, 1, 0, 0, 0, 1))
        tile_class : class, optional
            Class inheriting from `Tile` (default).
        tile_kwargs : dict, optional
            Key-word arguments for the tile initialisation (defaults to None).
        name_frmt : str, optional
            Formatter string for the tile name containing placeholders for two arguments, the mosaic row, and the
            mosaic col (defaults to "S{:03d}W{:03d}").
        **kwargs :
            Key-word arguments for the mosaic initialisation.

        Returns
        -------
        geospade.raster.RegularMosaicGeometry :
            Regular mosaic geometry defined by the given origin, tile sizes, number of tiles and the
            spatial reference system.

        """
        tile_kwargs = {} if tile_kwargs is None else tile_kwargs

        mosaic_ul_x, mosaic_ul_y = geotrans[0], geotrans[3]
        x_pixel_size, y_pixel_size = geotrans[1], geotrans[5]
        tile_height = int(round(y_tile_size / abs(y_pixel_size), DECIMALS))
        tile_width = int(round(x_tile_size / x_pixel_size, DECIMALS))
        # reset tile size to full units of pixel sizes
        x_tile_size = tile_width * x_pixel_size
        y_tile_size = tile_height * abs(y_pixel_size)
        # compute mosaic orientation
        ori = -np.arctan2(geotrans[2], geotrans[1])

        adjacency_matrix = np.zeros((n_rows, n_cols))
        tiles = []
        tile_counter = 0
        for mosaic_row in range(n_rows):
            # append points from N to S/ul to ll
            mosaic_row_ul_x, mosaic_row_ul_y = polar_point(mosaic_ul_x, mosaic_ul_y, y_tile_size*mosaic_row,
                                                           ori - np.pi / 2., False)
            for mosaic_col in range(n_cols):
                # append points from W to E/ul to ur
                tile_ul_x, tile_ul_y = polar_point(mosaic_row_ul_x, mosaic_row_ul_y, x_tile_size*mosaic_col,
                                                   ori, False)
                tile_geotrans = list(geotrans)
                tile_geotrans[0] = tile_ul_x
                tile_geotrans[3] = tile_ul_y
                tile_id = name_frmt.format(mosaic_row, mosaic_col)
                tile = tile_class(tile_height, tile_width, sref,
                                  geotrans=tuple(tile_geotrans),
                                  name=tile_id,
                                  **tile_kwargs)
                if boundary is not None:
                    if tile.within(boundary):
                        tile.mosaic_topology = "INNER"
                    elif tile.intersects(boundary):
                        tile.mosaic_topology = "BOUNDARY"
                    else:
                        tile.mosaic_topology = "OUTER"
                tiles.append(tile)
                adjacency_matrix[mosaic_row, mosaic_col] = tile_counter
                tile_counter += 1

        return cls(tiles, boundary=boundary, check_consistency=True, **kwargs)

    @property
    def shape(self):
        """
        2-tuple : Shape (height/number of tiles in Y direction, width/number of tiles in X direction)
        of the regular mosaic.

        """
        return self._adjacency_matrix.shape

    def xy2tile(self, x, y, sref=None):
        """
        Returns the tile intersecting with the given world system coordinates. If the coordinates are outside the
        mosaic boundary, no tile is returned.

        Parameters
        ----------
        x : number
            World system coordinate in X direction.
        y : number
            World system coordinate in Y direction.
        sref : SpatialRef, optional
            Spatial reference system of the world system coordinates. If None, the spatial reference system of the
            coordinates and the spatial reference system of the mosaic are assumed to be the same (default).

        Returns
        -------
        geospade.raster.Tile :
            Tile intersecting/matching with the given world system coordinates.

        """
        # create OGR point
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x, y)
        osr_sref = self.sref.osr_sref if sref is None else sref.osr_sref
        point.AssignSpatialReference(osr_sref)

        tile_oi = None
        if point.Intersects(self.boundary):  # point is inside grid definition
            mosaic_row, mosaic_col = self._raster_geom.xy2rc(x, y, sref=sref)
            tile_oi = self.__tile_from_rc(mosaic_row, mosaic_col)

        return tile_oi

    def get_neighbouring_tiles(self, tile_name, active_only=True, apply_mask=True):
        """
        Returns all tiles being in the direct neighbourhood of the tile with the given tile name.

        Parameters
        ----------
        tile_name : str
            Tile name.
        active_only : bool, optional
            If true, only active tiles are returned (default).
        apply_mask : bool, optional
            If true, tiles will have a mask for reflecting the relation with the exact mosaic boundary (default).

        Returns
        -------
        list of `Tile`:
            List of tiles being in the neighbourhood of `tile_name`.

        """
        cntr_tile = self.name2tile(tile_name)
        mosaic_row, mosaic_col = self._raster_geom.xy2rc(cntr_tile.ul_x, cntr_tile.ul_y)
        nbr_tiles = dict()

        for r in range(3):
            for c in range(3):
                if r == 1 and c == 1:  # exclude the main tile
                    continue
                nbr_tile_r = mosaic_row - 1 + r
                nbr_tile_c = mosaic_col - 1 + c
                if not self.__rc_is_within(nbr_tile_r, nbr_tile_c):  # ignore tiles being outside the grid boundaries
                    continue
                nbr_tile = self.__tile_from_rc(nbr_tile_r, nbr_tile_c)
                if nbr_tile is not None:
                    if active_only and not self._tiles.loc[nbr_tile.name, 'active']:
                        continue
                    if apply_mask:
                        nbr_tile = self._mask_tile(nbr_tile)
                    nbr_tiles[nbr_tile.name] = nbr_tile

        return nbr_tiles

    @_align_geom(align=True)
    def select_tiles_by_geom(self, geom, active_only=True, apply_mask=True):
        """
        Computes an intersection figure of the mosaic and another geometry and returns the tiles intersecting with this
        figure.

        Parameters
        ----------
        geom : ogr.Geometry
            Other geometry to intersect with.
        active_only : bool, optional
            If true, only active tiles are returned (default).
        apply_mask : bool, optional
            If true, tiles will have a mask for reflecting the relation with the exact mosaic boundary (default).

        Returns
        -------
        dict
            Dictionary of tiles (key=tilename, value=tile object).

        """

        intsctd_raster_geom = self._raster_geom.slice_by_geom(geom, True, False)
        if intsctd_raster_geom is None:
            return None
        intsctd_mosaic_height, intsctd_mosaic_width = intsctd_raster_geom.shape
        selected_tiles = dict()
        for intsctd_mosaic_row in range(intsctd_mosaic_height):
            for intsctd_mosaic_col in range(intsctd_mosaic_width):
                tile_ul_x, tile_ul_y = intsctd_raster_geom.rc2xy(intsctd_mosaic_row, intsctd_mosaic_col)
                mosaic_row, mosaic_col = self._raster_geom.xy2rc(tile_ul_x, tile_ul_y)
                tile = self.__tile_from_rc(mosaic_row, mosaic_col)
                if tile is not None:
                    if tile is not None:
                        if active_only and not self._tiles.loc[tile.name, 'active']:
                            continue
                        if apply_mask:
                            tile = self._mask_tile(tile)
                        selected_tiles[tile.name] = tile

        return selected_tiles

    def __tile_from_rc(self, r, c):
        """
        Retrieves tile from adjacency matrix indices, i.e. mosaic row and column numbers.

        Parameters
        ----------
        r : int
            Mosaic row number.
        c : int
            Mosaic column number.

        Returns
        -------
        geospade.raster.Tile :
            Tile related to the given mosaic indexes.

        """
        tile_idx = self._adjacency_matrix[r, c]
        tile = self._tiles.iloc[tile_idx]['tile']
        return tile

    def __rc_is_within(self, r, c):
        """
        Checks if a mosaic row and column index is within the boundaries of the mosaic.

        Parameters
        ----------
        r : int
            Row index.
        c : int
            Column index.

        Returns
        -------
        is_within : bool
            True if the given indexes are within the boundaries of the mosaic geometry, False otherwise.

        """
        max_mosaic_row, max_mosaic_col = self.shape
        is_within = True
        if r >= max_mosaic_row:
            is_within = False
        if c >= max_mosaic_col:
            is_within = False
        if r < 0:
            is_within = False
        if c < 0:
            is_within = False

        return is_within

    def __check_consistency(self, tiles):
        """
        Checks if the given tiles are consistent, which is the case if they match in pixel size, shape and
        spatial reference system.

        Parameters
        ----------
        tiles : list of geospade.raster.Tile
            List of tiles.

        Returns
        -------
        is_consistent : bool
            True if the mosaic is valid/consistent, False if not.

        """
        ref_tile = tiles[0]
        x_pixel_size = ref_tile.x_pixel_size
        y_pixel_size = ref_tile.y_pixel_size
        shape = ref_tile.shape
        sref = ref_tile.sref

        is_consistent = True
        for tile in tiles[1:]:
            if x_pixel_size != tile.x_pixel_size:
                is_consistent = False
            if y_pixel_size != tile.y_pixel_size:
                is_consistent = False
            if shape != tile.shape:
                is_consistent = False
            if sref != tile.sref:
                is_consistent = False
            if not is_consistent:
                break

        return is_consistent

    def __build_adjacency_matrix(self, tiles):
        """
        np.array : 2D numpy array representing an adjacency matrix, i.e. it contains information which tiles lay
        within the direct neighbourhood of one tile.

        """
        ref_tile = tiles[0]
        ul_xs, ul_ys = zip(*[(tile.ul_x, tile.ul_y) for tile in tiles])
        ll_xs, ll_ys = zip(*[tile.rc2xy(tile.n_rows - 1, 0, 'll') for tile in tiles])
        ur_xs, ur_ys = zip(*[tile.rc2xy(0, tile.n_cols - 1, 'ur') for tile in tiles])
        ll_x = min(ll_xs)
        ll_y = min(ll_ys)
        ul_x = min(ul_xs)
        ul_y = max(ul_ys)
        ur_x = max(ur_xs)
        ur_y = max(ur_ys)

        geotrans = ref_tile.geotrans
        sref = ref_tile.sref
        x_tile_size = ref_tile.x_size
        y_tile_size = ref_tile.y_size
        raster_geom_geotrans = (ul_x, x_tile_size, geotrans[2],
                                ul_y, geotrans[4], -y_tile_size)
        height = np.sqrt((ll_x - ul_x)**2 + (ll_y - ul_y)**2)
        width = np.sqrt((ul_x - ur_x)**2 + (ul_y - ur_y)**2)
        n_rows = int(round(height / y_tile_size, DECIMALS))
        n_cols = int(round(width / x_tile_size, DECIMALS))
        raster_geom = RasterGeometry(n_rows, n_cols, sref, raster_geom_geotrans)

        adjacency_matrix = np.ones(raster_geom.shape, dtype=int) * -1

        for i, tile in enumerate(tiles):
            x_cntr, y_cntr = tile.centre
            r, c = raster_geom.xy2rc(x_cntr, y_cntr)
            adjacency_matrix[r, c] = i

        return adjacency_matrix


if __name__ == '__main__':
    pass