import shapely
from shapely.geometry import Polygon
import math

class RasterGeometry:
    """
     Class which represents geometry of some georeferenced raster data. Basically
    it describes the extent of the data and the pixel grid. It also provides the
    information about the spatial reference of the data. The geometry can be
    used as a <emph>shapely</emph> geometry, or exported into
    <emph>cartopy</emph> projection.
    """

    def __init__(self, rows, cols, sref,
                 gt=(0, 1, 0, 0, 0, 1),
                 id=None,
                 description=None):
        """
        Basic constructor

        Parameters
        ----------
        rows : int
            Number of pixel rows
        cols : int
            Number of pixel columns
        sref : pyraster.spatial_ref.SpatilRef
            Spatial reference of the geometry
        gt : 6-tuple, optional
            GDAL Geotransform 'matrix'
        id : int, optional
            id of the geometry
        description : string, optional
            verbal description of the geometry

        """
        # Since the pixel_sizes, which are generally calculated from
        # geotransform can contain negative values, we have to make sure
        # that rows and cols are always positive integers
        # This causes practically any use of the abs function in this class
        # https://gis.stackexchange.com/questions/229780/why-is-gdalwarp-flipping-pixel-size-sign
        self.rows = abs(rows)
        self.cols = abs(cols)
        self.sref = sref
        self.gt = gt
        self.id = id
        self.description = description

        self.ori = -math.atan2(self.gt[2], self.gt[1])  # radians, usually 0

        # Width and height describe the dimensions of the geometry in the
        # units of the spatial reference e.g. meter or degrees
        self.height = self.rows * abs(self.y_pixel_size)
        self.width = self.cols * abs(self.x_pixel_size)

        # shapley representation
        self.geometry = Polygon(self.vertices)

    @classmethod
    def from_extent(cls, extent, sref, x_pixel_size, y_pixel_size):
        """
        Creates an RasterGeometry object from given extent (in units of the
        spatial reference) and pixel sizes in both pixel-grid directions. (=>
        pixel sizes determine the resolution of the data)

        Parameters
        ----------
        extent : 4-tuple
            coordinates defining extent from lower left to upper right corner
            (lower-left-x, lower-left-y, upper-right-x, upper-right-y )
        sref : pyraster.spatial_ref.SpatialRef
            Spatial reference of the geometry
        x_pixel_size : float
            Resolution in W-E direction
        y_pixel_size : float
            Resolution in N-S direction

        Returns
        -------
        RasterGeometry

        """

        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        # calculate upper-left corner for geotransform
        ul_x, ul_y = polar_point((ll_x, ll_y), height, math.pi / 2)
        gt = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
        # deal negative pixel sizes, hence the absolute value
        rows = abs(math.ceil(height / y_pixel_size))
        cols = abs(math.ceil(width / x_pixel_size))
        return RasterGeometry(rows, cols, sref, gt=gt)

    @classmethod
    def from_geometry(cls, geom, sref, x_pixel_size, y_pixel_size):
        """
        Creates a RasterGeometry object from a existing Shapley geometry.
        Since the RasterGeometry can represent rectangles only, non-rectangular
        shapely objects get converted into its bounding box. Since shapely
        geometries are not georeferenced, the  spatial reference has to be
        specified, as well as the resolution in both pixel grid directions

        Parameters
        ----------
        geom : shapely.geometry and its subclasses
            Shapely geometry from which the RasterGeometry object should be
            created
        sref : pyraster.spatial_ref.SpatialRef()
            Spatial reference of the both geometries
        x_pixel_size : float
            W-E pixel resolution
        y_pixel_size : float
            N-S pixel resloution

        Returns
        -------
        RasterGeometry

        """
        if len(geom.exterior.coords) == 5:  # This means actually 4 points
            # Assume that such polygon is a rotated rectangle
            # TODO: Check angles
            pts = list(geom.exterior.coords)[:-1]  # get coordinates
            # if we find index of the lower left corner, then we know
            # where to find other points such as upper right etc.
            ll_x = ll_y = float('inf')
            for i, pt in enumerate(pts):  # search for lower left corner
                newMin = pt[0] <= ll_x and pt[1] <= ll_y
                if newMin:
                    ll_idx = i
                    ll_x, ll_y = pt
            # we can determine other corners based on index the upper right
            # corner
            ur_x, ur_y = pts[ll_idx - 2]
            if pts[ll_idx - 1][0] < pts[ll_idx - 3][0]:
                lr_x, lr_y = pts[ll_idx - 3]
                ul_x, ul_y = pts[ll_idx - 1]
            else:
                lr_x, lr_y = pts[ll_idx - 1]
                ul_x, ul_y = pts[ll_idx - 3]

            # azimuth of the bottom base of the rectangle = orientation
            rot = math.atan2(lr_y - ll_y, lr_x - ll_x)

            gt = construct_geotransform((ul_x, ul_y),
                                        rot,
                                        (x_pixel_size, y_pixel_size),
                                        deg=False)

            width = math.hypot(lr_x - ll_x, lr_y - ll_y)
            height = math.hypot(ur_x - lr_x, ur_y - lr_y)
            rows = abs(math.ceil(height / y_pixel_size))
            cols = abs(math.ceil(width / x_pixel_size))

            return RasterGeometry(rows, cols, sref, gt=gt)

        else:
            # geom is not a rectangle
            bbox = geom.bounds
            return RasterGeometry.from_extent(bbox,
                                              sref,
                                              x_pixel_size,
                                              y_pixel_size)

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
        """ pixel size in the west - east direction"""
        return self.gt[1] / math.cos(self.ori)

    @property
    def y_pixel_size(self):
        """ pixel size in the north - south direction"""
        return self.gt[5] / math.cos(self.ori)

    @property
    def extent(self):
        return self.ll_x, self.ll_y, self.ur_x, self.ur_y

    @property
    def vertices(self):
        """
        A tuple containing all corners in following form
            (lower-left, lower-right, upper-right, upper-left)
        """
        return [
            (self.ll_x, self.ll_y),
            polar_point((self.ll_x, self.ll_y), self.width, self.ori),
            (self.ur_x, self.ur_y),
            polar_point((self.ll_x, self.ll_y), self.height, self.ori + math.pi / 2)
        ]

    # ? what are these properties for? They can be reached via the sref attribute
    @property
    def epsg(self):
        return self.sref.epsg

    @property
    def proj4(self):
        return self.sref.proj4

    # ? This might be even more confusing. One can think, that this proprety
    # contains is a wkt representation of the object e.g.:POLYGON(...)
    @property
    def wkt(self):
        return self.sref.wkt

    def intersect(self, other, snap_to_grid=True):
        """
        Computes a intersection figure of two geometries and returns its
        (grid axes-parallel rectangle) bounding box.

        Parameters
        ----------
        other : RasterGeoemetry or shapely.geometry
            other geometry
        snap_to_grid : bool
            if true, the computed corners of the intersection are rounded off to
            nearest pixel corner

        Returns
        -------
        RasterGeometry
            Bounding box of intersection figure

        """

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

    def to_cartopy_crs(self):
        bounds = (self.ll_x, self.ur_x, self.ll_y, self.ur_y)
        return self.sref.get_cartopy_crs(bounds=bounds)

    # Provisional version
    # def to_cartopy_crs(self):
    #     class CustomCCRS(ccrs.CRS): pass
    #
    #     crs = CustomCCRS(self.proj4)
    #     return crs

    def xy2rc(self, x, y):
        """
        Calculates pixel index in which given point in world system lies.
        Rounds to lower-closest integer
        Parameters
        ----------
        x :  float
            x coordinate
        y : list of float
            y coordinate

        Returns
        -------
        r , c : int
            row and col
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

    def rc2xy(self, r, c, center=False):
        """
        Returns the coordinates of the upper left corner of a pixel specified
        its by row and column . If center argument is set true, the function
        returns coordinates of the center of the pixel

        Parameters
        ----------
        r : int
            pixel row
        c : int
            pixel column
        center : bool
            determines, whether coordinates of upper left corner of the pixel
            should be returned, or coordinates of the center of the pixel

        Returns
        -------
        x, y : float
            spatial reference coordinates of the pixel
        """

        # geotransform 0 and 3 are coordinates of the upper left corner
        # ul corner of the pixel not the center!!!

        gt = self.gt
        x = gt[0] + c * gt[1] + r * gt[2]
        y = gt[3] + c * gt[4] + r * gt[5]

        # x = gt[0] + r * gt[1] + c * gt[2]
        # y = gt[3] + r * gt[4] + c * gt[5]

        # x = c // self.x_pixel_size
        # y = r // self.y_pixel_size

        if center:
            x += gt[1] / 2
            y += gt[5] / 2

        return x, y

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

    def buffer(self, val, unit='%'):
        """
        Builds an buffer around the geometry. The value can be specified in
        percent, pixels, or spatial reference units. Positive value extends,
        negative value shrinks the original object. Returns a new
        RasterGeometry object

        Parameters
        ----------
        val : float
            buffering value
        unit : string
            unit of val
            possible values:
                '%' : val is percentage of the geometry dimensions. (default)
                'px': val is number of pixels
                'sr': val is in spatial reference units (meters/degrees)
        Returns
        -------
        RasterGeometry

        Examples
        --------
        Shrink by 10 %:
        >>> geom = RandomRasterData.geometry  # random geometry
        >>> shrunk_geom = geom.buffer(-10)

        Grow by 10 px:
        >>> grown_geom = geom.buffer(10,unit='px')

        """
        if unit == '%':
            val /= 200  # =( ( val[%] / 100) / 2 )
            w = self.ur_x - self.ll_x
            h = self.ur_y - self.ll_y
            ll_x = self.ll_x - w * val
            ll_y = self.ll_y - h * val
            ur_x = self.ur_x + w * val
            ur_y = self.ur_y + h * val
        elif unit == 'px':
            pass
        elif unit == 'sr':
            pass
        else:
            pass

        return RasterGeometry.from_extent((ll_x, ll_y, ur_x, ur_y),
                                          self.sref,
                                          self.x_pixel_size,
                                          self.y_pixel_size)

    @classmethod
    def get_common_geometry(cls, geoms):
        # initial values
        ll_x = ll_y = math.inf
        ur_x = ur_y = -math.inf
        sref = geoms[0].sref
        x_pixel_size = geoms[0].x_pixel_size
        y_pixel_size = geoms[0].y_pixel_size

        # iterate over all geometries
        for g in geoms:
            # 1) get lowest - left-most corner
            if g.ll_x <= ll_x:
                if g.ll_y <= ll_y:
                    ll_x = g.ll_x
                    ll_y = g.ll_y
            # 2) get uppermost - right-most corner
            if g.ur_x <= ur_x:
                if g.ur_y <= ur_y:
                    ur_x = g.ur_x
                    ur_y = g.ur_y
            # 3) check if all geometries are compatible in sref and pixel_sizes
            if not g.sref == sref:
                raise ValueError('Error! Geometries have different Spatial '
                                 'Reference')
            if not g.x_pixel_size == x_pixel_size:
                raise ValueError('Error! Geometries have different pixel-sizes'
                                 ' in x dierection')
            if not g.y_pixel_size == y_pixel_size:
                raise ValueError('Error! Geometries have different pixel-sizes'
                                 ' in y dierection')


        extent = (ll_x, ll_y, ur_x, ur_y)

        return RasterGeometry.from_extent(extent,
                                          sref,
                                          x_pixel_size
                                          , y_pixel_size)

    def __contains__(self, point):
        """
        Checks whether a given coordinate is contained in this geometry

        Parameters
        ----------
        point : tuple
            point x-coordinate, point y-coordinate

        Returns
        -------
        x, y : tuple
            point x-coordinate, point y-coordinate

        """
        point = shapely.geometry.Point(*point)
        if self.geometry.contains(point):
            return True
        else:
            return False

    def __eq__(self, other):
        return self.vertices == other.vertices and \
               self.rows == other.rows and \
               self.cols == other.cols

    def __ne__(self, other):
        return not self == other

    __and__ = intersect

    def __repr__(self):
        return self.geometry.wkt

