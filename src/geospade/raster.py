import abc
import sys
import cv2
import ogr
import copy
import cartopy
import warnings
import numpy as np
import geopandas as geopd
import shapely
import shapely.wkt
from shapely import affinity
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as PolygonPatch

from geospade.tools import polar_point
from geospade.tools import is_rectangular
from geospade.tools import bbox_to_polygon
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
    OGR geometry is assumed to be given in first place!

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

            wrpd_geom = None
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
    The (boundary) geometry can be used as an OGR geometry, or can be exported into a Cartopy projection.
    """
    def __init__(self, n_rows, n_cols, sref,
                 geotrans=(0, 1, 0, 0, 0, 1),
                 geom_id=None,
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
        sref : geospade.spatial_ref.SpatialRef
            Instance representing the spatial reference of the geometry.
        geotrans : 6-tuple, optional
            GDAL geotransform tuple.
        geom_id : int or str, optional
            ID of the geometry.
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
        self.id = geom_id
        self.description = description
        self.parent = parent
        self.px_origin = px_origin

        # compute orientation of rectangle
        self.ori = -np.arctan2(self.geotrans[2], self.geotrans[1])  # radians, usually 0

        # compute boundary geometry
        boundary = Polygon(self.outer_boundary_corners)
        boundary_ogr = ogr.CreateGeometryFromWkt(boundary.wkt)
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
            Coordinates defining extent from lower left to upper right corner.
            (lower left x, lower left y, upper right x, upper right y)
        sref : geospade.spatial_ref.SpatialRef
            Spatial reference of the geometry/extent.
        x_pixel_size : float
            Absolute pixel size in x direction.
        y_pixel_size : float
            Absolute pixel size in y direction.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`,
            or `parent`.

        Returns
        -------
        RasterGeometry
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
        return cls(n_rows, n_cols, sref, geotrans=geotrans, px_origin='ul', **kwargs)

    @classmethod
    @_align_geom(align=True)
    def from_geometry(cls, geom, x_pixel_size, y_pixel_size, **kwargs):
        """
        Creates a `RasterGeometry` object from an existing geometry object.
        Since `RasterGeometry` can represent rectangles only, non-rectangular
        shapely objects get converted into its bounding box. Since, e.g. a `Shapely`
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
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`,
            or `parent`.

        Returns
        -------
        RasterGeometry
            Raster geometry object defined by the extent extent of the given geometry and the pixel sizes.

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
        raster_geoms : list of RasterGeometry
            List of `RasterGeometry` objects.

        Returns
        -------
        RasterGeometry
            Raster geometry containing all given raster geometries.

        Notes
        -----
        Note: All raster geometries must have the same spatial reference system and pixel sizes!

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
        """ float : Pixel size in y direction. """
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
        """ 4-tuple: Extent of the raster geometry with the pixel origins defined by the class
        (min_x, min_y, max_x, max_y). """
        return min([self.ll_x, self.ul_x]), min([self.ll_y, self.lr_y]), \
               max([self.ur_x, self.lr_x]), max([self.ur_y, self.ul_y])

    @property
    def outer_boundary_extent(self):
        """ 4-tuple: Outer extent of the raster geometry containing every pixel
        (min_x, min_y, max_x, max_y). """
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
        """ 2-tuple: Centre defined by the mass centre of the vertices. """
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
    def x_coords(self):
        """ np.ndarray : Returns all coordinates in x direction. """
        if self.is_axis_parallel:
            min_x, _ = self.rc2xy(0, 0)
            max_x, _ = self.rc2xy(0, self.n_cols - 1)
            return np.arange(min_x, max_x + self.x_pixel_size, self.x_pixel_size)
        else:
            cols = np.array(range(self.n_cols))
            return self.rc2xy(0, cols)[0]

    @property
    def y_coords(self):
        """ np.ndarray : Returns all coordinates in y direction. """
        if self.is_axis_parallel:
            _, min_y = self.rc2xy(self.n_rows - 1, 0)
            _, max_y = self.rc2xy(0, 0)
            return np.arange(max_y, min_y - self.y_pixel_size, -self.y_pixel_size)
        else:
            rows = np.array(range(self.n_rows))
            return self.rc2xy(rows, 0)[1]

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
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
        sref : SpatialRef, optional
            Spatial reference of the coordinates. Has to be given if the spatial
            reference is different than the spatial reference of the raster geometry.

        Returns
        -------
        bool :
            True if the specified x and y coordinates are located on the grid.

        """

        if sref is not None:
            x, y = transform_coords(x, y, sref, self.sref.osr_sref)

        x_is_on_grid = (x % self.x_pixel_size) <= (10**-DECIMALS)
        y_is_on_grid = (y % self.y_pixel_size) <= (10**-DECIMALS)

        return x_is_on_grid & y_is_on_grid

    @_align_geom(align=True)
    def intersects(self, other):
        """
        Evaluates if this `RasterGeometry` instance and another geometry intersect.

        Parameters
        ----------
        other : ogr.Geometry or RasterGeometry
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
        other : ogr.Geometry or RasterGeometry
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
        Evaluates if a geometry is in the raster geometry.

        Parameters
        ----------
        other : ogr.Geometry or RasterGeometry
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry is within the raster geometry, false if not.

        """
        return other.Within(self.boundary)

    @_align_geom(align=True)
    def slice_by_geom(self, other, snap_to_grid=True, inplace=False):
        """
        Computes an intersection figure of two geometries and returns its
        (grid axes-parallel rectangle) bounding box as a raster geometry.

        Parameters
        ----------
        other : ogr.Geometry or RasterGeometry
            Other geometry to intersect with.
        snap_to_grid : bool, optional
            If true, the computed corners of the intersection are rounded off to
            nearest pixel corner.
        inplace : bool
            If true, the current instance will be modified.
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        RasterGeometry
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
        intsct_raster_geom = RasterGeometry.from_extent(bbox, self.sref, self.x_pixel_size,
                                                        self.y_pixel_size, geom_id=self.id,
                                                        description=self.description, parent=self)

        if inplace:
            self.boundary = intsct_raster_geom.boundary
            self.n_rows = intsct_raster_geom.n_rows
            self.n_cols = intsct_raster_geom.n_cols
            self.geotrans = intsct_raster_geom.geotrans
            intsct_raster_geom = self

        return intsct_raster_geom

    def slice_by_rc(self, row, col, height=1, width=1, inplace=False):
        """
        Intersects raster geometry with a pixel extent.

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
            If false, a new `RasterGeometry` instance will be created.

        Returns
        -------
        RasterGeometry
            Raster geometry instance defined by the pixel extent.

        """
        intsct_raster_geom = None
        max_row = row + height
        max_col = col + width
        min_row, min_col, max_row, max_col = self.__align_pixel_extent(min_row=row, min_col=col,
                                                                       max_row=max_row, max_col=max_col)

        ul_x, ul_y = self.rc2xy(min_row, min_col, px_origin='ul')
        geotrans = build_geotransform(ul_x, ul_y, self.x_pixel_size, -self.y_pixel_size, 0)
        intsct_raster_geom = RasterGeometry(height, width, self.sref, geotrans, self.id,
                                            self.description, 'ul', self)

        if inplace:
            self.boundary = intsct_raster_geom.boundary
            self.n_rows = height
            self.n_cols = width
            self.geotrans = geotrans
            intsct_raster_geom = self

        return intsct_raster_geom

    def xy2rc(self, x, y, sref=None, px_origin="ul"):
        """
        Calculates an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
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

        c, r = xy2ij(x, y, self.geotrans, origin=px_origin)
        return r, c

    def rc2xy(self, r, c, px_origin="ul"):
        """
        Returns the coordinates of the center or a corner (dependent on ˋpx_originˋ) of a pixel specified
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

    def snap_to_grid(self, x, y, sref=None, px_origin="ul"):
        """
        Rounds the given world system coordinates `x` and `y` to a coordinate point of the grid spanned by the
        raster geometry. The coordinate anchor specified by `px_origin` is then returned.

        Parameters
        ----------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
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
            Raster geometry grid coordinate in x direction related to the origin defined by `px_origin`.
        new_y : float
            Raster geometry grid coordinate in y direction related to the origin defined by `px_origin`.

        """
        new_x, new_y = x, y
        if not self.is_raster_coord(x, y, sref=sref):
            row, col = self.xy2rc(x, y, sref=sref, px_origin="ul")
            new_x, new_y = self.rc2xy(row, col, px_origin=px_origin)

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
            Opacity of the boundary polygon (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes (default is None).
            If None, the projection of the spatial reference system of the raster geometry is taken.
        show : bool, optional
            If True, the plot result is shown (default is False).
        label_geom : bool, optional
            If True, the geometry ID is plotted at the center of the raster geometry (default is False).
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
        other_proj = None
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

        if self.id is not None and label_geom:
            transform = this_proj._as_mpl_transform(ax)
            ax.annotate(str(self.id), xy=self.centre, xycoords=transform, va="center", ha="center")

        if show:
            plt.show()

        return ax

    def scale(self, scale_factor, inplace=False):
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

        Returns
        -------
        RasterGeometry
            Scaled raster geometry.

        """

        return self.resize(scale_factor, unit='', inplace=inplace)

    def resize(self, buffer_size, unit='px', inplace=False):
        """
        Resizes the raster geometry. The resizing values can be specified as a scale factor,
        in pixels, or in spatial reference units. A positive value extends, a
        negative value shrinks the original object (except for the scaling factor).

        Parameters
        ----------
        buffer_size : number or list of numbers
            Buffering values, which have to be given in a clock-wise order, i.e. [left edge, top edge, right edge,
            bottom edge] or as one value.
        unit: string, optional
            Unit of the buffering value ˋbuffer_sizeˋ.
            Possible values are:
                '':	ˋvalˋ is given unitless as a positive scale factor with respect to edge length.
                'px':  ˋvalˋ is given as number of pixels.
                'sr':  ˋvalˋ is given in spatial reference units (meters/degrees).
        inplace : bool, optional
            If True, the current instance will be modified (default).
            If False, a new `RasterGeometry` instance will be created.

        Returns
        -------
        RasterGeometry
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

        res_raster_geom = None

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

        res_raster_geom = RasterGeometry.from_geometry(new_boundary_geom, self.x_pixel_size, self.y_pixel_size,
                                                       geom_id=self.id, description=self.description,
                                                       px_origin=self.px_origin, parent=self)

        if inplace:
            self.boundary = res_raster_geom.boundary
            self.n_rows = res_raster_geom.n_rows
            self.n_cols = res_raster_geom.n_cols
            self.geotrans = res_raster_geom.geotrans
            res_raster_geom = self

        return res_raster_geom

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
        max_row = max_row if max_row is not None else self.n_rows-1 # :(
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.n_rows-1, max_row)  # -1 because of Python indexing
        max_col = min(self.n_cols-1, max_col)  # -1 because of Python indexing
        if min_row > max_row:
            err_msg = "Row bounds [{};{}] exceed range of possible row indexes {}."
            raise ValueError(err_msg.format(min_row, max_row, self.n_rows-1))

        if min_col > max_col:
            err_msg = "Column bounds [{};{}] exceed range of possible column indexes {}."
            raise ValueError(err_msg.format(min_col, max_col, self.n_cols-1))

        return min_row, min_col, max_row, max_col

    @_align_geom(align=False)
    def __contains__(self, geom):
        """
        Checks whether a given geometry is contained in the raster geometry.

        Parameters
        ----------
        geom : RasterGeometry or ogr.Geometry
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

        return self.within(geom)

    def __eq__(self, other):
        """
        Checks if this and another raster geometry are equal.
        Equality holds true if the vertices, rows and columns are the same.

        Parameters
        ----------
        other: RasterGeometry
            Raster geometry object to compare with.

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
        other: RasterGeometry
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
        other: RasterGeometry or ogr.Geometry
            Raster geometry object to intersect with.

        Returns
        -------
        RasterGeometry
            Raster geometry instance defined by the bounding box of the intersection geometry.

        """
        geom_sref = other.GetSpatialReference()
        if not self.sref.osr_sref.IsSame(geom_sref):
            err_msg = "The spatial reference systems are not equal."
            raise ValueError(err_msg)

        return self.slice_by_geom(other, inplace=False)

    def __str__(self):
        """ str : String representation of a raster geometry as a Well Known Text (WKT) string. """

        return self.boundary_wkt

    def __getitem__(self, item):
        """
        Handles indexing of a raster geometry, which is herein defined as a 2D spatial indexing via x and y coordinates
        or pixel slicing.

        Parameters
        ----------
        item : 2-tuple or 3-tuple
            2-tuple: contains two indexes/slices for row and column pixel indexing/slicing.
            3-tuple: contains two coordinates/slices and one entry for a `SpatialRef` instance defining
                     the spatial reference system of the coordinates.

        Returns
        -------
        RasterGeometry
            Raster geometry defined by the intersection.

        """

        intsct_raster_geom = None
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
            extent = [min_f_idx, min_s_idx, max_f_idx - self.x_pixel_size, max_s_idx - self.y_pixel_size]
            intsct_raster_geom = self.from_extent(extent, sref, self.x_pixel_size, self.y_pixel_size,
                                                  geom_id=self.id, description=self.description, parent=self)
        else:
            err_msg = "The given way of indexing is not supported. Only two (pixel) or " \
                      "three (coordinates) indexes are supported."
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
        RasterGeometry

        """

        n_rows = self.n_rows
        n_cols = self.n_cols
        sref = copy.deepcopy(self.sref)
        geotrans = copy.deepcopy(self.geotrans)
        geom_id = self.id
        description = self.description
        px_origin = self.px_origin
        parent = self.parent

        return RasterGeometry(n_rows, n_cols, sref, geotrans, geom_id, description,
                              px_origin, parent)


class MosaicGeometry:
    """ Represents a homogeneous collection of `RasterGeometry` objects. """

    def __init__(self, raster_geoms, parent=None):
        """
        Constructor of `RasterGrid` class.

        Parameters
        ----------
        raster_geoms : geopandas.GeoDataFrame
            `DataFrame` with the column 'tile' containing raster geometries and their IDs as an index.
            In addition, the column 'geometry' includes the boundaries of the raster geometries as Shapely polygons.
        parent : MosaicGeometry, optional
            Parent mosaic geometry object.

        """

        self.tiles = raster_geoms
        self.parent = parent
        self.geom = RasterGeometry.from_raster_geometries(self.tiles['tile'])

    @property
    def geotrans(self):
        """ 6-tuple : Returns the GDAL geotransform parameters of the mosaic. """
        return self.geom.geotrans

    @property
    def sref(self):
        """ SpatialRef : Returns the spatial reference system of the mosaic. """
        return self.tiles['tile'][0].sref

    @property
    def orientation(self):
        """ float : Returns the mathematical positive orientation of the grid. """
        return self.tiles['tile'][0].ori

    @property
    def tile_ids(self):
        """  list : IDs of all raster geometries/tiles contained in the raster grid. """
        return sorted(list(self.tiles.index))

    @property
    def size(self):
        """ int : Number of pixels covered by the mosaic. """
        return sum([tile.size for tile in self.tiles['tile']])

    @property
    def outer_boundary_extent(self):
        """ 4-tuple : Outer extent of the mosiac containing all tiles and pixels (min_x, min_y, max_x, max_y). """

        return self.geom.outer_boundary_extent

    @classmethod
    def from_list(cls, raster_geoms, parent=None):
        """
        Creates a `MosaicGeometry` instance from a list of raster geometries.

        Parameters
        ----------
        raster_geoms : list of RasterGeometry
            List of raster geometry.

        Returns
        -------
        MosaicGeometry
            Mosaic defined by the given raster geometries.
        parent : MosaicGeometry, optional
            Parent mosaic geometry object.

        """
        tile_ids = [raster_geom.id for raster_geom in raster_geoms]
        boundaries = [raster_geom.boundary_shapely for raster_geom in raster_geoms]
        tiles = geopd.GeoDataFrame({'tile': raster_geoms, 'geometry': boundaries},
                                   index=tile_ids, crs=raster_geoms[0].sref.proj4)

        return cls(tiles, parent)

    @classmethod
    def from_dict(cls, raster_geoms_dict, parent=None):
        """
        Creates a `MosaicGeometry` instance from a dictionary of raster geometries.
        They keys of the dictionary are assumed to be the IDs of the tiles and the values
        are tiles, represented as raster geometries.

        Parameters
        ----------
        raster_geoms_dict : dict
            A dictionary containing raster geometries as values and their IDs as keys.
        parent : MosaicGeometry, optional
            Parent mosaic geometry object.

        Returns
        -------
        MosaicGeometry
            Mosaic defined by the given raster geometries.

        """
        tile_ids = list(raster_geoms_dict.keys())
        raster_geoms = list(raster_geoms_dict.values())
        boundaries = [raster_geom.boundary_shapely() for raster_geom in raster_geoms]
        tiles = geopd.GeoDataFrame({'tile': raster_geoms, 'geometry': boundaries},
                                   index=tile_ids, crs=raster_geoms[0].sref.proj4)

        return cls(tiles, parent)

    def tile_from_id(self, tile_id):
        """ RasterGeometry : Returns raster geometry according to the given tile ID. """
        return self.tiles.loc[tile_id]['tile']

    def neighbours_from_id(self, tile_id):
        """
        list of RasterGeometry : Collects the neighbouring raster geometries/tiles
        according to the given tile ID.
        """
        idxs = self.tiles.touches(self.tiles.loc[tile_id]['geometry'])
        return list(self.tiles[idxs]['tile'])

    def intersection_by_coords(self, x, y, sref=None, snap_to_grid=True, inplace=False):
        """
        Intersects a geometry with the raster grid and returns a cropped raster grid.
        Thereby, each raster geometry gets cropped to.

        Parameters
        ----------
        x : number
            X coordinate of the world coordinate system.
        y : number
            Y coordinate of the world coordinate system.
        sref : SpatialRef, optional
            Spatial reference system of the coordinates. Has to be given if the
            spatial reference system of the coordinates differs from the spatial
            reference system of the mosaic (defauls to None).
        snap_to_grid : bool, optional
            If True, the computed corners of the intersection are rounded off to
            nearest pixel corner (defaults to True).
        inplace : bool, optional
            If True, the current instance will be modified.
            If False, a new `MosaicGeometry` instance will be created (default).

        Returns
        -------
        MosaicGeometry
            Cropped mosaic geometry with cropped tiles/raster geometries.

        """

        poi = ogr.Geometry(ogr.wkbPoint)
        poi.AddPoint(x, y)

        osr_sref = sref.osr_sref if sref is not None else self.sref.osr_sref
        poi.AssignSpatialReference(osr_sref)

        return self.intersection_by_geom(poi, snap_to_grid, inplace)

    @_align_geom(align=True)
    def intersection_by_geom(self, geom, snap_to_grid=True, inplace=False):
        """
        Intersects a geometry with the raster grid and returns a cropped raster grid.
        Thereby, each raster geometry gets cropped to.

        Parameters
        ----------
        geom : ogr.Geometry
            Other geometry to intersect with.
        snap_to_grid : bool, optional
            If True, the computed corners of the intersection are rounded off to
            nearest pixel corner (defaults to True).
        inplace : bool, optional
            If True, the current instance will be modified.
            If False, a new `MosaicGeometry` instance will be created (default).

        Returns
        -------
        MosaicGeometry
            Cropped mosaic geometry with cropped tiles/raster geometries.

        """

        if not self.geom.intersects(geom):
            wrn_msg = "Geometry does not intersect with raster grid."
            warnings.warn(wrn_msg)
        else:
            intersct_mosaic_geom = None
            idxs = self.tiles.intersects(shapely.wkt.loads(geom.ExportToWkt()))
            tiles = copy.deepcopy(self.tiles[idxs])
            intersection = lambda x: x.slice_by_geom(geom, snap_to_grid, inplace)
            boundary = lambda x: shapely.wkt.loads(x.boundary.ExportToWkt())
            tiles['tile'] = tiles['tile'].apply(intersection)
            tiles['geometry'] = tiles['tile'].apply(boundary)

            intersct_mosaic_geom = MosaicGeometry(tiles, parent=self)

            if inplace:
                self.geom = intersct_mosaic_geom.geom
                self.parent = self
                self.tiles = tiles
                intersct_mosaic_geom = self

            return intersct_mosaic_geom

    def plot(self, ax=None, facecolor='tab:red', edgecolor='black', edgewidth=1, alpha=1., proj=None,
             show=False, label_tiles=False, add_country_borders=True, extent=None):
        """
        Plots the mosaic on a map, i.e. all of its tiles/raster geometries.

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
            Opacity of the tile boundary polygon (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes.
        show : bool, optional
            If True, the plot result is shown (default is False).
        label_tiles : bool, optional
            If True, the tile ID is plotted at the center of the tile (default is False).
        add_country_borders : bool, optional
            If True, country borders are added to the plot (`cartopy.feature.BORDERS`) (default is False).
        extent : tuple or list, optional
            Coordinate/Map extent of the plot, given as [min_x, min_y, max_x, max_y]
            (default is None, meaning global extent).

        Returns
        -------
        matplotlib.pyplot.axes
            Matplotlib axis containing a Cartopy map with the plotted mosaic.

        """

        for i in range(len(self.tiles)):
            ax = self.tiles['tile'][i].plot(ax, facecolor, edgecolor, edgewidth, alpha,
                                            proj, show, label_tiles, add_country_borders,
                                            extent)

        return ax

    def __len__(self):
        """ int : Returns the number of tiles in the mosaic. """
        return len(self.tiles)

    def __str__(self):
        """ str : String representation of a mosaic as a GeoPandas data frame. """
        return str(self.tiles)

    def __getitem__(self, item):
        """
        Handles indexing of a mosaic, which is herein defined as either 1D indexing with a tile ID or
        3D spatial indexing via x and y coordinates and their spatial reference.

        Parameters
        ----------
        item : str or int or 3-tuple
            str or int: referring to tile IDs.
            3-tuple: contains two coordinates/slices and one entry for a `SpatialRef` instance defining
                     the spatial reference system of the coordinates.

        Returns
        -------
        MosaicGeometry or RasterGeometry
            Returns either a mosaic geometry for coordinate intersection or a raster geometry for
            tile indexing.

        """

        ret_geom = None

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

        if isinstance(item, (str, int)):
            if item in self.tile_ids:
                ret_geom = self.tile_from_id(item)
            else:
                err_msg = "Tile ID '{}' is not available.".format(item)
                raise KeyError(err_msg)
        elif len(item) == 3:
            sref = item[2]
            extent = [(min_f_idx, min_s_idx), (max_f_idx, max_s_idx)]
            boundary = bbox_to_polygon(extent, sref)
            ret_geom = self.intersection_by_geom(boundary)
        else:
            err_msg = "The given way of indexing is not supported. Only one (tile ID) or " \
                      "three (coordinates) indexes are supported."
            raise ValueError(err_msg)

        return ret_geom