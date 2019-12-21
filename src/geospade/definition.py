import os
import osr
import ogr
import math
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

from geospade.spatial_ref import SpatialRef

# TODO: rename operations

def _any_geom2ogr_geom(func):
    """
    A decorator which converts an input geometry (first argument) into an OGR.geometry object
    """

    def wrapper(*args, **kwargs):
        sref = kwargs.get("sref", None)
        if isinstance(sref, SpatialRef):
            sref = sref.to_osr()
        ogr_geom = any_geom2ogr_geom(args[0], osr_sref=sref)
        if len(args) > 1:
            return func(ogr_geom, **kwargs)
        else:
            args = args[1:]
            return func(ogr_geom, *args, **kwargs)

    return wrapper

class SwathGeometry(object):
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

        pass


class RasterGeometry(object):
    """
    Represents the geometry of a georeferenced raster.
    It describes the extent and the grid of the raster along with its spatial reference.
    The (boundary) geometry can be used as an OGR geometry, or can be exported into a Cartopy projection.
    """
    # TODO: add segment_size test and None behaviour to doku
    def __init__(self, rows, cols, sref,
                 gt=(0, 1, 0, 0, 0, 1),
                 geom_id=None,
                 description=None,
                 segment_size=None):
        """
        Constructor of the `RasterGeometry` class.

        Parameters
        ----------
        rows: int
            Number of pixel rows.
        cols: int
            Number of pixel columns.
        sref: geospade.spatial_ref.SpatialRef
            Spatial reference of the geometry.
        gt: 6-tuple, optional
            GDAL Geotransform tuple.
        geom_id: int or str, optional
            ID of the geometry.
        description : string, optional
            Verbal description of the geometry.
        segment_size: float, optional
            For precision: distance in input units of longest segment of the geometry polygon.
        """

        self.rows = rows
        self.cols = cols
        self.sref = sref
        self.gt = gt
        self.geom_id = geom_id
        self.description = description

        self.ori = -np.arctan2(self.gt[2], self.gt[1])  # radians, usually 0

        # shapely representation
        boundary = Polygon(self.vertices)
        boundary_ogr = ogr.CreateGeometryFromWkt(boundary.wkt)
        boundary_ogr.AssignSpatialReference(sref.to_osr())
        self.geometry = segmentize_geometry(boundary_ogr, segment=segment_size)

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
            (lower-left-x, lower-left-y, upper-right-x, upper-right-y)
        sref : geospade.spatial_ref.SpatialRef
            Spatial reference of the geometry/extent.
        x_pixel_size : float
            Resolution in x direction.
        y_pixel_size : float
            Resolution in y direction.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description` or `segment_size`.

        Returns
        -------
        `RasterGeometry`
            Raster geometry object defined by the given extent and pixel sizes.
        """

        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        # calculate upper-left corner for geotransform
        ul_x, ul_y = polar_point((ll_x, ll_y), height, math.pi / 2.)
        gt = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
        # deal negative pixel sizes, hence the absolute value
        rows = abs(math.ceil(height / y_pixel_size))
        cols = abs(math.ceil(width / x_pixel_size))
        return cls(rows, cols, sref, gt=gt, **kwargs)

    # TODO: add function for deriving rectangular properties from a list of points
    # TODO: exchange math with numpy
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
        geom: shapely.geometry or ogr.geometry
            Geometry object from which the RasterGeometry object should be
            created
        x_pixel_size: float
            Resolution in x direction
        y_pixel_size: float
            Resolution in y direction
        sref: geospade.spatial_ref.SpatialRef, optional
            Spatial reference of the geometry object.
            Has to be given if the spatial reference cannot be derived from 'geom'.
        **kwargs
            Keyword arguments for `RasterGeometry` constructor, i.e. `geom_id`, `description` or `segment_size`.

        Returns
        -------
        RasterGeometry
        """

        #geom, geom_type, sref = other2shapely_geom(geom, sref=sref)
        geom = shapely.wkt.loads(geom.ExportToWKT())
        geom_ch = geom.convex_hull

        if is_rectangular(geom_ch) :  # This means the polygon can be described directly as a RasterGeometry
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
            rot = math.atan2(lr_y - ll_y, lr_x - ll_x)

            # create GDAL geotransform
            gt = construct_geotransform((ul_x, ul_y), rot, (x_pixel_size, y_pixel_size), deg=False)

            # define raster properties
            width = math.hypot(lr_x - ll_x, lr_y - ll_y)
            height = math.hypot(ur_x - lr_x, ur_y - lr_y)
            rows = abs(math.ceil(height / y_pixel_size))
            cols = abs(math.ceil(width / x_pixel_size))

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
        raster_geoms: list RasterGeometry objects
                        List of RasterGeometry objects.

        Returns
        -------
        RasterGeometry
                        Raster geometry containing all given raster geometries given by ˋraster_geomsˋ.
        """

        # first geometry serves as a reference


        sref = raster_geoms[0].sref
        x_pixel_size = raster_geoms[0].x_pixel_size
        y_pixel_size = raster_geoms[0].y_pixel_size

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
        return self.ori == 0.

    @property
    def ll_x(self):
        """ x coordinate of lower left corner """

        x, _ = self.rc2xy(self.rows, 0)
        return x

    @property
    def ll_y(self):
        """ y coordinate of lower left corner """

        _, y = self.rc2xy(self.rows, 0)
        return y

    @property
    def ur_x(self):
        """ x coordinate of upper right corner """

        x, _ = self.rc2xy(0, self.cols)
        return x

    @property
    def ur_y(self):
        """ y coordinate of upper right corner """

        _, y = self.rc2xy(0, self.cols)
        return y

    @property
    def x_pixel_size(self):
        """ Pixel size in x direction """

        return self.gt[1]

    @property
    def y_pixel_size(self):
        """ Pixel size in y direction """

        return self.gt[5]

    @property
    def h_pixel_size(self):
        """ Pixel size in W-E direction """

        return self.x_pixel_size / math.cos(self.ori)

    @property
    def v_pixel_size(self):
        """ Pixel size in N-S direction """

        return self.y_pixel_size / math.cos(self.ori)

    @property
    def width(self):
        """ Width of the raster geometry """
        return self.cols * abs(self.x_pixel_size)

    @property
    def height(self):
        """ Height of the raster geometry """
        return self.rows * abs(self.y_pixel_size)

    @property
    def extent(self):
        return self.ll_x, self.ll_y, self.ur_x, self.ur_y

    @property
    def area(self):
        return self.width * self.height

    @property
    def vertices(self):
        """
        4-list of 2-tuples: A tuple containing all corners in the following form
        (lower-left, lower-right, upper-right, upper-left).
        """

        vertices= [(self.ll_x, self.ll_y),
                   polar_point((self.ll_x, self.ll_y), self.width, self.ori),
                   (self.ur_x, self.ur_y),
                   polar_point((self.ll_x, self.ll_y), self.height, self.ori + math.pi / 2),
                   (self.ll_x, self.ll_y)]
        return vertices


    def to_wkt(self):
        """
        Returns WKT representation of the boundary of a `RasterGeometry`.

        Returns
        -------
        str
            WKT representation of the boundary of a `RasterGeometry`.
        """

        return self.geometry.ExportToWKT()


    def intersects(self, other):
        """
        Evaluates if this geometry and another geometry intersect.

        Parameters
        ----------

        other: geospade.geometry.RasterGeometry or shapely.geometry or ogr.Geometry
                                            Other geometry to evaluate an intersection against.

            Returns
            -------

            intersects: bool
                    True if both geometries intersect, false if not.
        """

        if issubclass(other, RasterGeometry):
            geometry = other.geometry
        else:
            geometry = any_geom2ogr_geom(other)

        return self.geometry.intersects(geometry)


    def touches(self, other):
        """
        Evaluates if this geometry and another geometry touch each other.

        Parameters
        ----------

        other: geospade.geometry.RasterGeometry or shapely.geometry or ogr.Geometry
                                            Other geometry to evaluate an intersection against.

            Returns
            -------

            intersects: bool
                    True if both geometries intersect, false if not.
        """

        if issubclass(type(other), geospade.geometry.RasterGeometry):
            touch = self.geometry.touch(other.geometry)
        else:
            other = other2shapely_geom(other)
        touch = self.geometry.touch(other)

        return touch


    def intersection(self, other, snap_to_grid=True):
        """
        Computes an intersection figure of two geometries and returns its
        (grid axes-parallel rectangle) bounding box.

        Parameters
        ----------
        other: RasterGeometry or shapely.geometry
            Other geometry.
        snap_to_grid: bool, optional
            If true, the computed corners of the intersection are rounded off to
            nearest pixel corner.

        Returns
        -------
        RasterGeometry
            Bounding box of intersection geometry.

        """
        if not self.intersects(other):
            return None

        if issubclass(type(other), shapely.geometry.base.BaseGeometry):
            # other is shapely Polygon, LineString, Circle...
            isect = self.geometry.intersection(other)
        else:
            # other is RasterGeometry object
            isect = self.geometry.intersection(other.geometry)

        bbox = isect.bounds
        if snap_to_grid:
            ll_px = self.xy2rc(bbox[0], bbox[1])
            ur_px = self.xy2rc(bbox[2], bbox[3])
            bbox = self.rc2xy(*ll_px) + self.rc2xy(*ur_px)

        return RasterGeometry.from_extent(bbox,
                                          sref=self.sref,
                                          x_pixel_size=self.x_pixel_size,
                                          y_pixel_size=self.y_pixel_size)

    # TODO: set this bounds
    def to_cartopy_crs(self, bounds=None):
        bounds = (self.ll_x, self.ur_x, self.ll_y, self.ur_y)
        return self.sref.get_cartopy_crs(bounds=bounds)


    def xy2rc(self, x, y):
        """
        Calculates an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x: float
            x coordinate.
        y: float
            y coordinate.

        Returns
        -------
        r: int
            Pixel row number.
        c: int
                        Pixel column number.

        Notes
        -----
        Rounds to the closest, lower integer.
        """

        gt = self.gt

        c = (-1.0 * (gt[2] * gt[3] - gt[0] * gt[5] + gt[5] * x - gt[2] * y) /
             (gt[2] * gt[4] - gt[1] * gt[5]))
        r = (-1.0 * (-1 * gt[1] * gt[3] + gt[0] * gt[4] - gt[4] * x + gt[1] * y) /
             (gt[2] * gt[4] - gt[1] * gt[5]))

        # round to lower-closest integer
        c = math.floor(c)
        r = math.floor(r)

        return int(r), int(c)


    def rc2xy(self, r, c, origin="ll"):
        """
        Returns the coordinates of the center or a corner (dependend on ˋoriginˋ) of a pixel specified
        by a row and column number.

        Parameters
        ----------
        r: int
            Pixel row number.
        c: int
            Pixel column number.
        origin: str
            Determines the world system origin of the pixel. It can be "upperleft", "upperright", "lowerright", "lowerleft" or "center".

        Returns
        -------
        x: float
            x world system coordinate.
        y: float
            y world system coordinate.
        """

        # geotransform 0 and 3 are coordinates of the upper left corner

        gt = self.gt

        px_shift_map = {"ul": (0, 0),
                        "ur": (1, 0),
                        "lr": (1, 1),
                        "ll": (0, 1),
                        "c": (.5, .5)}

        if origin in px_shift_map.keys():
            px_shift = px_shift_map[origin]
        else:
            user_wrng = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead".format(origin)
            raise Warning(user_wrng)
            px_shift = (0, 0)

        c += px_shift[0]
        r += px_shift[1]
        x = gt[0] + c * gt[1] + r * gt[2]
        y = gt[3] + c * gt[4] + r * gt[5]

        return x, y

    # TODO: plot raster geometry boundary
    # TODO: use matplotlib as an option
    # probably not needed. Maybe for debugging purposes
    #
    # def plot(self, ax=None, color='tab:blue', alpha=1, grid=False, proj=None):
    #     trafo = self.to_cartopy_crs()
    #     if ax is None:
    #         if proj is None:
    #             proj = ccrs.Mollweide()
    #         ax = plt.axes(projection=proj)
    #         ax.set_global()
    #         ax.gridlines()
    #         ax.coastlines()
    #
    #     patch = PolygonPatch(self.vertices, color=color, alpha=alpha,
    #                          transform=trafo, zorder=0)
    #     ax.add_patch(patch)
    #
    #     if grid:
    #         pass
    #         # TODO Still needs some work
    #         # patches = []
    #         # ll, ur, ul = [self.vertices[0]] + self.vertices[-2:]
    #         #
    #         # # row dividers:
    #         # for r in range(self.rows + 1):
    #         #     start = polar_point(ul, r * self.x_pixel_size, self.ori)
    #         #     end = polar_point(ll, r * self.x_pixel_size, self.ori)
    #         #     path = Path((start, end))
    #         #     patch = PathPatch(path,
    #         #                       color='black',
    #         #                       )
    #         #     patches.append(patch)
    #         # # col dividers
    #         # for c in range(self.cols + 1):
    #         #     start = polar_point(ul, c * self.y_pixel_size, self.ori - math.pi / 2)
    #         #     end = polar_point(ur, c * self.x_pixel_size, self.ori - math.pi / 2)
    #         #     path = Path((start, end))
    #         #     patch = PathPatch(path,
    #         #                       color='black',
    #         #                       zorder=1000)
    #         #     patches.append(patch)
    #         #
    #         # patches = PatchCollection(patches,
    #         #                           transform=trafo,
    #         #                           match_original=True)
    #         # ax.add_collection(patches)
    #
    #     return ax

    def scale(self, scale_factor):
        return self.resize(scale_factor, unit='')

    # TODO: list of values for each border or one value
    def resize(self, buffer_size, unit='px', in_place=True):
        """
        Resizes the raster geometry. The value can be specified in
        percent, pixels, or spatial reference units. A positive value extends, a
        negative value shrinks the original object.

        Parameters
        ----------
        buffer_size: number or list of numbers
            Buffering value.
        unit: string, optional
            Unit of the buffering value ˋvalˋ.
            Possible values are:
                '':	ˋvalˋ is given unitless as a scale factor
                'px':  ˋvalˋ is given as number of pixels.
                'sr':  ˋvalˋ is given in spatial reference units (meters/degrees).
        Returns
        -------
        RasterGeometry

        Examples
        --------
        Shrink by 10 %:
        raster_geom = RasterGeometry(...)  # random geometry
        raster_geom.resize(-10, unit='%')

        Grow by 10 px:
        raster_geom.resize(10, unit='px')
        """

        if isinstance(buffer_size, (float, int)):
            buffer_size = [buffer_size]*4

        if unit not in ['', 'px', 'sr']:
            err_msg = "Unit '{}' is unknown. Please use 'px', 'sr' or ''."
            raise Exception(err_msg.format(unit))

        # first, convert the geometry to a shapely geometry
        geometry = shapely.wkt.loads(self.geometry.ExportToWKT())
        # then, rotate the geometry to be axis parallel if it is not axis parallel
        geometry = affinity.rotate(geometry.convex_hull, -self.ori, 'center')
        # loop over all edges
        scale_factors = []
        for i in range(len(buffer_size)):
            buffer_size_i = buffer_size[i]
            if (i == 1) or (i == 3):  # left edge
                if unit == '':
                    scale_factor = buffer_size_i
                elif unit == 'px':
                    scale_factor = buffer_size_i/self.rows
                else:
                    scale_factor = buffer_size_i/self.height
            else:
                if unit == '':
                    scale_factor = buffer_size_i
                elif unit == 'px':
                    scale_factor = buffer_size_i/self.cols
                else:
                    scale_factor = buffer_size_i/self.width

            scale_factors.append(scale_factor)

        vertices = list(geometry.exterior.coords)
        xs = [vertice[0] for vertice in vertices]
        ys = [vertice[1] for vertice in vertices]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        res_min_x = min_x - self.width * scale_factors[0]
        res_min_y = min_y - self.height * scale_factors[3]
        res_min_x = min_x - self.width * scale_factors[0]
        res_min_y = min_y - self.height * scale_factors[3]
        # dx_0 = self.width * scale_factors[0]
        # dy_0 = self.height * scale_factors[1]
        # alpha_0 = np.arctan(dy_0/dx_0)
        # d_0 = np.sqrt(dx_0**2 + dy_0**2)
        # ul_x, ul_y =
        # polar_point(self.rc2xy(0, 0), d_0, alpha_0)
        # ul_x -= self.width * scale_factors[0] * np.cos(self.ori)
        # ul_y += self.height * scale_factors[1] * np.cos(self.ori)
        # ur_x = self.ur_x + self.width * scale_factors[2] * np.tan(self.ori)
        # ur_y = self.ur_y + self.width * scale_factors[1] * np.cos(self.ori)
        # lr_x, lr_y = self.rc2xy(self.rows, self.cols)
        # lr_x -= self.width * scale_factors[2] * np.tan(self.ori)
        # lr_y += self.height * scale_factors[3] * np.tan(self.ori)
        # ll_x = self.ll_x + self.width * scale_factors[0] * np.tan(self.ori)
        # ll_y = self.ll_y + self.width * scale_factors[3] * np.cos(self.ori)

        if in_place:
            return self
        else:
            return RasterGeometry.from_extent((ll_x, ll_y, ur_x, ur_y),
                                                 self.sref,
                                                 self.x_pixel_size,
                                                 self.y_pixel_size)

    # TODO: same sref, same pixel, move it to toolbox
    def get_rel_extent(self, other, unit='px'):
        """
        Computes extent in relative pixel or world system coordinates with respect to a second `RasterGeometry` object.

        Parameters
        ----------
        other: RasterGeometry
            `RasterGeometry` defining the origin.
        unit: string, optional
            Unit of the relative coordinates.
            Possible values are:
                'px':  Relative coordinates are given as the number of pixels.
                'sr':  Relative coordinates are given as spatial reference units (meters/degrees).
        Returns
        -------
        4-tuple of numbers
            Relative extent of this raster geometry with respect to the other raster geometry ((min_x, min_y, max_x, max_y)).
        """

        other_ul = other.rc2xy(0, 0)
        rel_extent = (self.extent[0] - other_ul[0],  self.extent[1] - other_ul[1],
                      self.extent[2] - other_ul[0], self.extent[3] - other_ul[1])
        if unit == 'sr':
            return rel_extent
        elif unit == 'px':
            return (int(round(rel_extent[0]/self.x_pixel_size)), int(round(rel_extent[1]/self.y_pixel_size)),
                    int(round(rel_extent[2]/self.x_pixel_size)), int(round(rel_extent[3]/self.y_pixel_size)))
        else:
            err_msg = "Unit {} is unknown. Please use 'px' or 'sr'."
            raise Exception(err_msg.format(unit))

    #TODO: call OGR within
    def __contains__(self, geom):
        """
        Checks whether a given point is contained in the raster geometry.

        Parameters
        ----------
        point: tuple, shapely.geometry.Point, ogr.Geometry.Point
            Point to check.

        Returns
        -------
        bool
            True if the given point is within the raster geometry, otherwise false.
        """
        if isinstance(point, tuple):
            point = shapely.geometry.Point(*point)
        elif isinstance(point, ogr.Geometry.Point):
            point = shapely.geometry.from_wkt(point.ExportToWKT)

        if self.geometry.contains(point):
            return True
        else:
            return False


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
        bool:
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
        other: RasterGeometry
                        Raster geometry object to compare with.

        Returns
        -------
        bool:
                    True if both raster geometries are the not the same, otherwise false.
        """


        return not self == other


    def __and__(self, other):
        """
        And operation intersects both raster geometries.

        Parameters
        ----------
        other: RasterGeometry
                        Raster geometry object to intersect with.

        Returns
        -------
        RasterGeometry
    Bounding box of intersection geometry.
        """

        return self.intersection(other)


    def __repr__(self):
        """
        String representation of a raster geometry as WKT string.

        Returns
        -------
        str:
                        WKT string representing the raster geometry.
        """


        return self.geometry.wkt


class RasterGrid():
    def __init__(self, raster_geoms, adjacency_map=None, map_type=None):
        self.geoms = raster_geoms
        self.geom_ids = [raster_geom.geom_id for raster_geom in raster_geoms]
        self.map_type = map_type
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
                row = [self.geoms[i]]
            else:
                row.append(self.geoms[i])

        return np.array(adjacency_list)

    def __build_adjacency_dict(self):
        n = len(self.geoms)
        adjacency_dict = dict()
        for i in range(n):
            neighbours = []
            for j in range(n):
                if self.geoms[i].touches(self.geoms[j]):
                    neighbours.append(self.geoms[j])
            adjacency_dict[i] = neighbours

        return adjacency_dict

    def _get_raster_geom(self, geom_id):
        idx = self.geom_ids.index(geom_id)
        return self.geoms[idx]

    def neighbours(self, raster_geom):
        idx = self.geom_ids.index(raster_geom.id)
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

    def intersection(self, geom_roi, raster_geom=None):
        if geom_roi is not None:
            geoms = [geom_roi]
        else:
            geoms = copy.deepcopy(self.geoms)

        geoms_intersected = []
        while len(geoms) > 0:
            geoms_new = []
            for geom in geoms:
                if not (geom.intersects(geom_roi) or geom.within(geom_roi)):
                    continue
                else:
                    geoms_intersected.append(geom)
                    geoms_add.extend(self.neighbours(geom))

            geoms = geoms_new

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_raster_geom(key)
        elif isinstance(key, tuple):
            x_min =
            y_min =
            x_max =
            y_max =
            p_1 = (x_min, y_min)
            p_2 = (x_min, y_max)
            p_3 = (x_max, y_max)
            p_4 = (x_max, y_min)
            geom_roi = Polygon([p_1, p_2, p_3, p_4, p_1])
            return self.intersection(geom_roi)