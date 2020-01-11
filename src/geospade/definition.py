import os
import abc
import sys
import osr
import ogr
import copy
import shapely
from shapely import affinity

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, LineString
from matplotlib.patches import Polygon as PolygonPatch, Path, PathPatch

from geospade.operation import polar_point
from geospade.operation import segmentize_geometry
from geospade.operation import any_geom2ogr_geom
from geospade.operation import construct_geotransform
from geospade.operation import is_rectangular
from geospade.operation import xy2ij
from geospade.operation import ij2xy
from geospade.operation import bbox_to_polygon

from geospade.spatial_ref import SpatialRef


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

    def wrapper(*args, **kwargs):
        sref = kwargs.get("sref", None)
        if isinstance(sref, SpatialRef):
            sref = sref.osr_sref

        geom = args[0]
        if isinstance(geom, RasterGeometry):
            geom = geom.geometry

        ogr_geom = any_geom2ogr_geom(geom, osr_sref=sref)
        if len(args) > 1:
            return f(ogr_geom, **kwargs)
        else:
            args = args[1:]
            return f(ogr_geom, *args, **kwargs)

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


class RasterGeometry:
    """
    Represents the geometry of a georeferenced raster.
    It describes the extent and the grid of the raster along with its spatial reference.
    The (boundary) geometry can be used as an OGR geometry, or can be exported into a Cartopy projection.
    """

    def __init__(self, rows, cols, sref,
                 gt=(0, 1, 0, 0, 0, 1),
                 geom_id=None,
                 description=None,
                 segment_size=None,
                 parent=None):
        """
        Constructor of the `RasterGeometry` class.

        Parameters
        ----------
        rows : int
            Number of pixel rows.
        cols : int
            Number of pixel columns.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Instance representing the spatial reference of the geometry.
        gt : 6-tuple, optional
            GDAL geotransform tuple.
        geom_id : int or str, optional
            ID of the geometry.
        description : string, optional
            Verbal description of the geometry.
        segment_size : float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
            If None, only the corner points are used for creating the boundary geometry.
        parent : RasterGeometry
            Parent `RasterGeometry` instance.
        """

        self.rows = rows
        self.cols = cols
        self.gt = gt
        self.id = geom_id
        self.description = description
        self.parent = parent
        self._segment_size = segment_size

        # get internal spatial reference representation from osr object
        if isinstance(sref, osr.SpatialReference):
            sref = SpatialRef.from_osr(sref)
        self.sref = sref

        # compute orientation of rectangle
        self.ori = -np.arctan2(self.gt[2], self.gt[1])  # radians, usually 0

        # compute boundary geometry
        boundary = Polygon(self.vertices)
        boundary_ogr = ogr.CreateGeometryFromWkt(boundary.wkt)
        boundary_ogr.AssignSpatialReference(self.sref.osr_sref)
        self.boundary = segmentize_geometry(boundary_ogr, segment=segment_size)

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
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`, `segment_size` or
            `parent`.

        Returns
        -------
        RasterGeometry
            Raster geometry object defined by the given extent and pixel sizes.
        """

        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        # calculate upper-left corner for geotransform
        ul_x, ul_y = polar_point((ll_x, ll_y), height, np.pi / 2.)
        gt = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
        # deal negative pixel sizes, hence the absolute value
        rows = abs(np.ceil(height / y_pixel_size))
        cols = abs(np.ceil(width / x_pixel_size))
        return cls(rows, cols, sref, gt=gt, **kwargs)

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
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description`, `segment_size` or
            `parent`.

        Returns
        -------
        RasterGeometry
        """

        geom = shapely.wkt.loads(geom.ExportToWKT())
        geom_ch = geom.convex_hull

        if is_rectangular(geom_ch):  # This means the polygon can be described directly as a RasterGeometry
            geom_pts = list(geom.exterior.coords)  # get boundary coordinates
            # separate coordinates
            xs = [geom_pt[0] for geom_pt in geom_pts]
            ys = [geom_pt[1] for geom_pt in geom_pts]
            # find corner coordinates
            ul_x = min(xs)
            ul_y = max(ys)
            lr_x = max(xs)
            lr_y = min(ys)
            ur_x = lr_x
            ur_y = ul_y
            ll_x = ul_x
            ll_y = lr_y

            # azimuth of the bottom base of the rectangle = orientation
            rot = np.arctan2(lr_y - ll_y, lr_x - ll_x)

            # create GDAL geotransform
            gt = construct_geotransform((ul_x, ul_y), rot, (x_pixel_size, y_pixel_size), deg=False)

            # define raster properties
            width = np.hypot(lr_x - ll_x, lr_y - ll_y)
            height = np.hypot(ur_x - lr_x, ur_y - lr_y)
            rows = abs(np.ceil(height / y_pixel_size))
            cols = abs(np.ceil(width / x_pixel_size))

            return RasterGeometry(rows, cols, sref, gt=gt, **kwargs)
        else:
            # geom is not a rectangle
            bbox = geom.bounds
            return cls.from_extent(bbox, sref, x_pixel_size, y_pixel_size, **kwargs)

    @classmethod
    def get_common_geometry(cls, raster_geoms):
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

        # define function for checking if all given geometries are consistent, i.e. if they have the same pixel size
        # and spatial reference.
        def check_consistency(raster_geom):
            if raster_geom.sref != sref:
                raise ValueError('Geometries have a different spatial reference.')

            if raster_geom.x_pixel_size != x_pixel_size:
                raise ValueError('Geometries have different pixel-sizes in x direction.')

            if raster_geom.y_pixel_size != y_pixel_size:
                raise ValueError('Geometries have different pixel-sizes in y direction.')

        _ = map(raster_geoms, lambda raster_geom: check_consistency(raster_geom))

        ll_xs = []
        ll_ys = []
        ur_xs = []
        ur_ys = []
        for raster_geom in raster_geoms:
            ll_xs.append(raster_geom.ll_x)
            ll_ys.append(raster_geom.ll_y)
            ur_xs.append(raster_geom.ur_x)
            ur_ys.append(raster_geom.ur_y)

        extent = (min(ll_xs), min(ll_ys), max(ur_xs), max(ur_ys))

        return cls.from_extent(extent, sref, x_pixel_size, y_pixel_size)

    @property
    def is_axis_parallel(self):
        """ bool : True if the `RasterGeometry` is not rotated , i.e. it is axis-parallel. """
        return self.ori == 0.

    @property
    def ll_x(self):
        """ float : x coordinate of the lower left corner. """
        x, _ = self.rc2xy(self.rows, 0)
        return x

    @property
    def ll_y(self):
        """ float: y coordinate of the lower left corner. """
        _, y = self.rc2xy(self.rows, 0)
        return y

    @property
    def ur_x(self):
        """ float : x coordinate of the upper right corner. """
        x, _ = self.rc2xy(0, self.cols)
        return x

    @property
    def ur_y(self):
        """ float : y coordinate of the upper right corner. """
        _, y = self.rc2xy(0, self.cols)
        return y

    @property
    def x_pixel_size(self):
        """ float : Pixel size in x direction. """
        return self.gt[1]

    @property
    def y_pixel_size(self):
        """ float : Pixel size in y direction. """

        return self.gt[5]

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
        return self.cols * abs(self.x_pixel_size)

    @property
    def height(self):
        """ float : Height of the raster geometry. """
        return self.rows * abs(self.y_pixel_size)

    @property
    def extent(self):
        """ 4-tuple: Extent of the raster geometry. """
        return self.ll_x, self.ll_y, self.ur_x, self.ur_y

    @property
    def area(self):
        """ float : Area covered by the raster geometry. """
        return self.width * self.height

    @property
    def vertices(self):
        """
        4-list of 2-tuples : A tuple containing all corners in the following form (lower left, lower right,
        upper right, upper left).
        """

        vertices = [(self.ll_x, self.ll_y),
                   polar_point((self.ll_x, self.ll_y), self.width, self.ori),
                   (self.ur_x, self.ur_y),
                   polar_point((self.ll_x, self.ll_y), self.height, self.ori + np.pi / 2),
                   (self.ll_x, self.ll_y)]
        return vertices

    def to_wkt(self):
        """
        Returns Well Known Text (WKT) representation of the boundary of a `RasterGeometry`.

        Returns
        -------
        str
        """

        return self.boundary.ExportToWKT()

    @_any_geom2ogr_geom
    def intersects(self, other):
        """
        Evaluates if this `RasterGeometry` instance and another geometry intersect.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to evaluate an intersection with.

        Returns
        -------
        bool
            True if both geometries intersect, false if not.
        """

        return self.boundary.Intersects(other)

    @_any_geom2ogr_geom
    def touches(self, other):
        """
        Evaluates if this `RasterGeometry` instance and another geometry touch each other.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to evaluate a touch operation with.

        Returns
        -------
        bool
            True if both geometries touch each other, false if not.
        """

        touch = self.boundary.Touches(other)

        return touch

    @_any_geom2ogr_geom
    def intersection(self, other, snap_to_grid=True, segment_size=None, inplace=True):
        """
        Computes an intersection figure of two geometries and returns its
        (grid axes-parallel rectangle) bounding box.

        Parameters
        ----------
        other : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry to intersect with.
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
        if not self.intersects(other):
            return None

        intersection = self.boundary.Intersection(other)
        bbox = intersection.bounds
        if snap_to_grid:
            ll_px = self.xy2rc(bbox[0], bbox[1])
            ur_px = self.xy2rc(bbox[2], bbox[3])
            bbox = self.rc2xy(*ll_px) + self.rc2xy(*ur_px)

        if segment_size is None:
            segment_size = self._segment_size

        intsct_raster_geom = RasterGeometry.from_extent(bbox, sref=self.sref, x_pixel_size=self.x_pixel_size,
                                                        y_pixel_size=self.y_pixel_size, geom_id=self.id,
                                                        description=self.description, segment_size=segment_size,
                                                        parent=self)

        if inplace:
            self._segment_size = segment_size
            self.boundary = intsct_raster_geom.boundary
            self.rows = intsct_raster_geom.rows
            self.cols = intsct_raster_geom.cols
            self.gt = intsct_raster_geom.gt
            return self
        else:
            return intsct_raster_geom

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
            bounds = self.extent
        return self.sref.get_cartopy_crs(bounds=bounds)

    def xy2rc(self, x, y):
        """
        Calculates an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.

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
        c, r = xy2ij(x, y, self.gt)
        return r, c

    def rc2xy(self, r, c, origin="ll"):
        """
        Returns the coordinates of the center or a corner (dependend on ˋoriginˋ) of a pixel specified
        by a row and column number.

        Parameters
        ----------
        r : int
            Pixel row number.
        c : int
            Pixel column number.
        origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur", default)
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

        return ij2xy(c, r, self.gt, origin=origin)

    def plot(self, ax=None, color='tab:red', alpha=1., proj=None, show=False):
        """
        Plots the boundary of the raster geometry on a map.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        color : str, optional
            Color code as described at https://matplotlib.org/3.1.0/tutorials/colors/colors.html (default is 'tab:red').
        alpha : float, optional
            Opacity of the boundary polygon (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes.
        show : bool, optional
            If true, the plot result is shown (default is False).

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
            #ax.set_global()
            ax.gridlines()
            ax.coastlines()

        boundary = shapely.wkt.loads(self.boundary.ExportToWKT())
        patch = PolygonPatch(list(boundary.exterior.coords), color=color, alpha=alpha,
                             transform=trafo, zorder=0)
        ax.add_patch(patch)

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
        boundary = shapely.wkt.loads(self.boundary.ExportToWKT())
        # then, rotate the geometry to be axis parallel if it is not axis parallel
        boundary = affinity.rotate(boundary.convex_hull, -self.ori, 'center')
        # loop over all edges
        scale_factors = []
        for i in range(len(buffer_size)):
            buffer_size_i = buffer_size[i]
            if (i == 0) or (i == 2):  # left and right edge
                if unit == '':
                    scale_factor = buffer_size_i
                elif unit == 'px':
                    scale_factor = buffer_size_i/(self.rows*0.5)  # 0.5 because buffer size always refers to half the edge length
                else:
                    scale_factor = buffer_size_i/(self.height*0.5)
            else:
                if unit == '':
                    scale_factor = buffer_size_i
                elif unit == 'px':
                    scale_factor = buffer_size_i/(self.cols*0.5)
                else:
                    scale_factor = buffer_size_i/(self.width*0.5)

            scale_factors.append(scale_factor)

        # get extent
        vertices = list(boundary.exterior.coords)
        xs = [vertice[0] for vertice in vertices]
        ys = [vertice[1] for vertice in vertices]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        # resize extent (0.5 because buffer size always refers to half the edge length)
        res_min_x = min_x - self.width * scale_factors[0] * 0.5
        res_min_y = min_y - self.height * scale_factors[3] * 0.5
        res_max_x = max_x + self.width * scale_factors[1] * 0.5
        res_max_y = max_y + self.height * scale_factors[2] * 0.5

        # create new boundary geometry and rotate it back
        new_boundary = Polygon(((res_min_x, res_min_y),
                                (res_min_x, res_max_y),
                                (res_max_x, res_max_y),
                                (res_max_x, res_min_y),
                                (res_min_x, res_min_y)))
        new_boundary = affinity.rotate(new_boundary, self.ori, 'center')

        if segment_size is None:
            segment_size = self._segment_size

        res_raster_geom = RasterGeometry.from_geometry(new_boundary, self.x_pixel_size, self.y_pixel_size, self.sref,
                                                       segment_size=segment_size, description=self.description,
                                                       geom_id=self.id, parent=self)

        if inplace:
            self._segment_size = segment_size
            self.boundary = res_raster_geom.boundary
            self.rows = res_raster_geom.rows
            self.cols = res_raster_geom.cols
            self.gt = res_raster_geom.gt
            return self
        else:
            return res_raster_geom

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

        return self.boundary.Within(geom)

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
               self.rows == other.rows and \
               self.cols == other.cols

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

        return self.intersection(other)

    def __repr__(self):
        """
        String representation of a raster geometry as a Well Known Text (WKT) string.

        Returns
        -------
        str
            WKT string representing the raster geometry.
        """

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

            extent = [min_x, min_y, max_x, max_y]
            boundary = bbox_to_polygon(extent, osr_sref=self.sref.osr_sref, segment=segment_size)

            return self.intersection(boundary, segment_size=segment_size, inplace=False)


class RasterGrid(metaclass=abc.ABCMeta):
    def __init__(self, raster_geoms, adjacency_map=None, map_type=None, parent=None):
        self.geoms = raster_geoms
        self._geom_ids = [raster_geom.geom_id for raster_geom in raster_geoms]
        self.map_type = map_type
        self.parent = parent

        if adjacency_map is None:
            if map_type == "dict":
                self.adjacency_map = self.__build_adjacency_dict()
            elif map_type == "mat":
                self.adjacency_map = self.__build_adjacency_matrix()
            elif map_type == "array":
                self.adjacency_map = self.__build_adjacency_array()
            else:
                err_msg = "Map type '{}' is unknown.".format(map_type)
                raise Exception(err_msg)
        else:
            self.adjacency_map = adjacency_map

        # get spatial reference info from first raster geometry
        self.sref = self.geoms[0].sref
        self.ori = self.geoms[0].ori

    @property
    def ids(self):
        if self._geom_ids is None:
            self._geom_ids = [raster_geom.geom_id for raster_geom in self.geoms]
        return self._geom_ids

    def __build_adjacency_matrix(self):
        n = len(self.geoms)
        i_idxs = []
        j_idxs = []
        for i in range(n):
            for j in range(i, n):
                if self.geoms[i].touches(self.geoms[j]):
                    i_idxs.extend([i, j])
                    j_idxs.extend([j, i])

        adjacency_matrix = np.zeros((n, n))
        adjacency_matrix[i_idxs, j_idxs] = 1

        return adjacency_matrix

    def __build_adjacency_array(self):
        adjacency_list = []
        row = [self.geoms[0]]
        for i in range(1, self.geoms):
            if not self.geoms[i].touches(self.geoms[i - 1]):
                adjacency_list.append(row)
                row = [i]
            else:
                row.append(i)

        return np.array(adjacency_list)

    def __build_adjacency_dict(self):
        n = len(self.geoms)
        adjacency_dict = dict()
        for i in range(n):
            neighbours = []
            for j in range(n):
                if i == j:
                    continue
                if self.geoms[i].touches(self.geoms[j]):
                    neighbours.append(self.geoms[j])
            adjacency_dict[i] = neighbours

        return adjacency_dict

    def _get_raster_geom(self, geom_id):
        idx = self.ids.index(geom_id)
        return self.geoms[idx]

    def neighbours(self, raster_geom):
        idx = self.ids.index(raster_geom.id)
        if self.map_type == "dict":
            return self.adjacency_map[idx]
        elif self.map_type == "matrix":
            idxs = np.where(self.adjacency_map[idx, :] == 1)
            return [self.geoms[idx] for idx in idxs]
        elif self.map_type == "array":
            map_idx = np.where(self.adjacency_map == idx)
            i = map_idx[0]
            j = map_idx[1]
            map_idxs = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
                        (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1)]
            n, m = self.adjacency_map.shape
            neighbours = []
            for map_idx in map_idxs:
                i = map_idx[0]
                j = map_idx[1]

                if i < 0:
                    i = n + i
                elif i > (n - 1):
                    i = i - n

                if j < 0:
                    j = m + j
                elif j > (m - 1):
                    j = j - m

                neighbours.append(self.geoms[self.adjacency_map[i, j]])

            return neighbours

    @_any_geom2ogr_geom
    @abc.abstractmethod
    def intersection(self, other, inplace=True, **kwargs):
        if self.map_type == "array":
            geoms_intersected, adjacency_map = self._array_intersection(other, **kwargs)
        else:
            adjacency_map = None
            geoms_intersected = self._simple_intersection(other, **kwargs)

        raster_grid = RasterGrid(geoms_intersected, adjacency_map=adjacency_map, map_type=self.map_type, parent=self)
        if inplace:
            self.adjacency_map = adjacency_map
            self._geom_ids = None
            self.geoms = geoms_intersected
            return self
        else:
            return raster_grid

    def _simple_intersection(self, other, **kwargs):
        geoms_intersected = []
        for geom in self.geoms:
            intersection = geom.intersection(other, inplace=False, **kwargs)
            if intersection is not None:
                geoms_intersected.append(intersection)
                geoms_add = self.neighbours(geom)
                while len(geoms_add) > 0:
                    geoms_add_new = []
                    for geom_add in geoms_add:
                        intersection = geom_add.intersection(other, inplace=False, **kwargs)
                        if intersection is None:
                            continue
                        else:
                            geoms_intersected.append(intersection)
                            geoms_add_new.extend(self.neighbours(geom_add))
                    geoms_add = geoms_add_new
                break
        return geoms_intersected

    def _array_intersection(self, other, **kwargs):

        # create RasterGeometry, which describes the array
        rows, cols = np.where(~np.isnan(self.adjacency_map))
        raster_geom_u = self.geoms[min(rows)]
        raster_geom_l = self.geoms[min(cols)]
        gt = construct_geotransform((raster_geom_l.ul_x, raster_geom_u.ul_y), self.ori,
                                    (raster_geom_u.width, raster_geom_u.height), deg=False)
        array_geom = RasterGeometry(self.adjacency_map.shape[0], self.adjacency_map.shape[1], self.sref,
                                    gt)
        # transform the given geometry to the spatial reference of the grid
        other = other.TransformTo(self.sref)
        extent = other.GetEnvelope()
        # extract boundaries and adjacency array
        ll_r, ll_c = array_geom.xy2rc(extent[0], extent[1])
        ur_r, ur_c = array_geom.xy2rc(extent[2], extent[3])
        adjacency_map_roi = self.adjacency_map[ur_r:(ll_r + 1), ll_c:(ur_c + 1)]
        # get intersected raster geometries
        raster_geoms_roi = self.geoms[adjacency_map_roi.flatten()]
        raster_geoms_roi = raster_geoms_roi[~np.isnan(raster_geoms_roi)]
        # recreate adjacency array
        adjacency_map_list = list(range(adjacency_map_roi.shape[0]*adjacency_map_roi.shape[1]))
        adjacency_map = np.array(adjacency_map_list).reshape(adjacency_map_roi.shape)

        return raster_geoms_roi, adjacency_map

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_raster_geom(item)
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

                extent = [min_x, min_y, max_x, max_y]
                boundary = bbox_to_polygon(extent, osr_sref=self.sref.osr_sref, segment=segment_size)

                return self.intersection(boundary, segment_size=segment_size, inplace=False)
        else:
            err_msg = "Key is only allowed to be of type str or tuple."
            KeyError(err_msg)
