import unittest
import random
import shapely
import shapely.wkt
import numpy as np
import cartopy.crs as ccrs
from shapely import affinity
from shapely.geometry import Polygon

from geospade.spatial_ref import SpatialRef
from geospade.definition import RasterGeometry
from geospade.definition import RasterGrid


class RasterGeometryTest(unittest.TestCase):
    """ Tests functionalities of `RasterGeometry`. """

    def setUp(self):

        # set up spatial reference system
        self.sref = SpatialRef(4326)

        # define region of interest/extent
        ll_x = random.randrange(-50., 50., 10.)
        ll_y = random.randrange(-50., 50., 10.)
        ur_x = ll_x + random.randrange(10., 50., 10.)
        ur_y = ll_y + random.randrange(10., 50., 10.)
        self.geom = Polygon((
            (ll_x, ll_y),
            (ur_x, ll_y),
            (ur_x, ur_y),
            (ll_x, ur_y)
        ))
        self.extent = tuple(map(float, (ll_x, ll_y, ur_x, ur_y)))
        self.x_pixel_size = 0.5
        self.y_pixel_size = -0.5
        self.raster_geom = RasterGeometry.from_extent(self.extent, self.sref, self.x_pixel_size, self.y_pixel_size)

        # create rotated raster geometry
        geom_nap = affinity.rotate(self.geom, 45, 'center')
        self.raster_geom_rot = RasterGeometry.from_geometry(geom_nap, self.x_pixel_size, self.y_pixel_size,
                                                            sref=self.sref)

    def test_from_extent(self):
        """ Tests setting up a raster geometry from a given extent. """

        self.assertTupleEqual(self.raster_geom.extent, self.extent)

    def test_from_geom(self):
        """ Tests setting up a raster geometry from a given geometry. """

        raster_geom = RasterGeometry.from_geometry(self.geom, self.x_pixel_size, self.y_pixel_size, sref=self.sref)

        self.assertListEqual(raster_geom.vertices, list(self.geom.exterior.coords))

    def test_get_common_geom(self):
        """ Tests the creation of an encasing raster geometry from multiple raster geometries. """

        # create different raster geometries
        extent_a = self.extent
        extent_b = tuple(np.array(extent_a) - 2)  # shift extent negatively
        extent_c = tuple(np.array(extent_a) + 2)  # shift extent positively
        # create raster geometries from different extents
        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_extent(extent_b, self.sref, self.x_pixel_size, self.y_pixel_size)
        raster_geom_c = RasterGeometry.from_extent(extent_c, self.sref, self.x_pixel_size, self.y_pixel_size)
        raster_geom_d = RasterGeometry.from_extent(extent_a, self.sref, self.x_pixel_size*2, self.y_pixel_size*2)

        # resize the main geometry
        raster_geom_a.scale(0.5)

        # create common raster geometry from the first three raster geometries
        raster_geom = RasterGeometry.from_raster_geometries([raster_geom_a, raster_geom_b, raster_geom_c])
        ll_x, ll_y = raster_geom_b.rc2xy(raster_geom_b.n_rows - 1, 0, px_origin="ll")
        ur_x, ur_y = raster_geom_c.rc2xy(0, raster_geom_b.n_cols - 1, px_origin="ur")
        new_extent = (ll_x, ll_y, ur_x, ur_y)

        self.assertTupleEqual(raster_geom.extent, new_extent)

        # test if error is raised, when raster geometries with different resolutions are joined
        try:
            raster_geom = RasterGeometry.from_raster_geometries([raster_geom_a, raster_geom_b, raster_geom_c,
                                                              raster_geom_d])
        except ValueError:
            assert True

    def test_parent(self):
        """ Tests accessing of parent `RasterData` objects. """

        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_scaled, inplace=False)

        assert raster_geom_intsct.parent == self.raster_geom

        # tests finding the parent root
        raster_geom_scaled = raster_geom_scaled.scale(0.5, inplace=False)
        raster_geom_intsct = raster_geom_intsct.intersection_by_geom(raster_geom_scaled, inplace=False)

        assert raster_geom_intsct.parent_root == self.raster_geom

    def test_is_axis_parallel(self):
        """ Tests if geometries are axis parallel. """

        assert self.raster_geom.is_axis_parallel
        assert not self.raster_geom_rot.is_axis_parallel

    def test_pixel_size(self):
        """ Check raster and vertical/horizontal pixel size. """

        assert self.raster_geom.x_pixel_size == self.raster_geom.h_pixel_size
        assert self.raster_geom.y_pixel_size == self.raster_geom.v_pixel_size
        assert self.raster_geom_rot.x_pixel_size != self.raster_geom_rot.h_pixel_size
        assert self.raster_geom_rot.y_pixel_size != self.raster_geom_rot.v_pixel_size

    def test_area(self):
        """ Checks computation of the area covered by the raster geometry. """

        # compute area of the extent
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]
        extent_area = extent_height*extent_width

        assert extent_area == self.raster_geom.area

    def test_vertices(self):
        """ Tests if extent nodes are equal to the vertices of the raster geometry. """

        # create vertices from extent
        vertices = [(self.extent[0], self.extent[1]),
                    (self.extent[2], self.extent[1]),
                    (self.extent[2], self.extent[3]),
                    (self.extent[0], self.extent[3]),
                    (self.extent[0], self.extent[1])]

        self.assertListEqual(self.raster_geom.vertices, vertices)

    def test_x_coords(self):
        """ Tests coordinate retrieval along x dimension. """

        # x coordinate retrieval for axis-parallel raster geometry
        assert len(self.raster_geom.x_coords) == self.raster_geom.n_cols
        assert self.raster_geom.x_coords[-1] == self.raster_geom.rc2xy(0, self.raster_geom.n_cols - 1)[0]
        assert self.raster_geom.x_coords[0] == self.raster_geom.rc2xy(0, 0)[0]
        rand_idx = random.randrange(1, self.raster_geom.n_cols - 2, 1)
        assert self.raster_geom.x_coords[rand_idx] == self.raster_geom.rc2xy(0, rand_idx)[0]

        # x coordinate retrieval for rotated raster geometry (rounding introduced because of machine precision)
        assert len(self.raster_geom_rot.x_coords) == self.raster_geom_rot.n_cols
        assert round(self.raster_geom_rot.x_coords[-1], 1) == \
               round(self.raster_geom_rot.rc2xy(0, self.raster_geom_rot.n_cols - 1)[0], 1)
        assert round(self.raster_geom_rot.x_coords[0], 1) == \
               round(self.raster_geom_rot.rc2xy(0, 0)[0], 1)
        rand_idx = random.randrange(1, self.raster_geom_rot.n_cols - 2, 1)
        assert round(self.raster_geom_rot.x_coords[rand_idx], 1) == \
               round(self.raster_geom_rot.rc2xy(0, rand_idx)[0], 1)

    def test_y_coords(self):
        """ Tests coordinate retrieval along x dimension. """

        # x coordinate retrieval for axis-parallel raster geometry
        assert len(self.raster_geom.y_coords) == self.raster_geom.n_rows
        assert self.raster_geom.y_coords[-1] == self.raster_geom.rc2xy(self.raster_geom.n_rows - 1, 0)[1]
        assert self.raster_geom.y_coords[0] == self.raster_geom.rc2xy(0, 0)[1]
        rand_idx = random.randrange(1, self.raster_geom.n_rows - 2, 1)
        assert self.raster_geom.y_coords[rand_idx] == self.raster_geom.rc2xy(rand_idx, 0)[1]

        # x coordinate retrieval for rotated raster geometry (rounding introduced because of machine precision)
        assert len(self.raster_geom_rot.y_coords) == self.raster_geom_rot.n_rows
        assert round(self.raster_geom_rot.y_coords[-1], 1) == \
               round(self.raster_geom_rot.rc2xy(self.raster_geom_rot.n_rows - 1, 0)[1], 1)
        assert round(self.raster_geom_rot.y_coords[0], 1) == \
               round(self.raster_geom_rot.rc2xy(0, 0)[1], 1)
        rand_idx = random.randrange(1, self.raster_geom_rot.n_rows - 2, 1)
        assert round(self.raster_geom_rot.y_coords[rand_idx], 1) == \
               round(self.raster_geom_rot.rc2xy(rand_idx, 0)[1], 1)

    def test_intersection(self):
        """ Test intersection with different geometries. """

        # test intersection with own geometry
        raster_geom_intsct = self.raster_geom.intersection_by_geom(self.raster_geom.boundary, inplace=False)
        assert self.raster_geom == raster_geom_intsct

        # test intersection with scaled geometry
        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_scaled, inplace=False)

        assert raster_geom_scaled == raster_geom_intsct

        # test intersection with partial overlap
        extent_shifted = tuple(np.array(self.extent) + 2)
        extent_intsct = (extent_shifted[0], extent_shifted[1], self.extent[2], self.extent[3])
        raster_geom_shifted = RasterGeometry.from_extent(extent_shifted, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_shifted, inplace=False)

        assert raster_geom_intsct.extent == extent_intsct

        # test intersection with no overlap
        extent_no_ovlp = (self.extent[2] + 1., self.extent[3] + 1.,
                          self.extent[2] + 5., self.extent[3] + 5.)
        raster_geom_no_ovlp = RasterGeometry.from_extent(extent_no_ovlp, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_no_ovlp, inplace=False)

        assert raster_geom_intsct is None

    def test_intersection_snap_to_grid(self):
        """ Tests intersection with and without the `snap_to_grid` option. """

        # create raster geometry which has a slightly smaller extent
        raster_geom_reszd = self.raster_geom.resize(-abs(self.x_pixel_size)/5., unit='sr', inplace=False)

        # execute intersection with 'snap_to_grid=True'
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_reszd, snap_to_grid=True, inplace=False)
        assert raster_geom_intsct == self.raster_geom

        # execute intersection with 'snap_to_grid=False'
        raster_geom_intsct = self.raster_geom.intersection_by_geom(raster_geom_reszd, snap_to_grid=False, inplace=False)
        assert raster_geom_intsct != self.raster_geom

    def test_segment_size(self):
        """ Tests intersections with different segment sizes. """
        pass

    def test_touches(self):
        """ Tests if raster geometries touch each other. """

        # create raster geometry which touches the previous one
        extent_tchs = (self.extent[2], self.extent[3],
                       self.extent[2] + 5., self.extent[3] + 5.)
        raster_geom_tchs = RasterGeometry.from_extent(extent_tchs, self.sref,
                                                      self.x_pixel_size, self.y_pixel_size)
        assert self.raster_geom.touches(raster_geom_tchs)

        # create raster geometry which does not touch the previous one
        extent_no_tchs = (self.extent[2] + 1., self.extent[3] + 1.,
                       self.extent[2] + 5., self.extent[3] + 5.)
        raster_geom_no_tchs = RasterGeometry.from_extent(extent_no_tchs, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        assert not self.raster_geom.touches(raster_geom_no_tchs)

    def test_to_cartopy_crs(self):
        """ Tests creation of a Cartopy CRS from a raster geometry. """
        pass

    def test_coord_conversion(self):
        """ Tests bijective coordinate conversion. """

        r_0 = random.randint(0, self.raster_geom.n_rows)
        c_0 = random.randint(0, self.raster_geom.n_cols)

        x, y = self.raster_geom.rc2xy(r_0, c_0)
        r, c = self.raster_geom.xy2rc(x, y)

        self.assertTupleEqual((r_0, c_0), (r, c))

    def test_plot(self):
        """ Tests plotting function of a raster geometry. """

        self.raster_geom.plot(proj=self.raster_geom.to_cartopy_crs())

        # test plotting with labelling
        self.raster_geom.id = "E048N018T1"
        self.raster_geom.plot(proj=ccrs.PlateCarree(), label_geom=True)

    def test_scale(self):
        """ Tests scaling of a raster geometry. """

        # helper variables
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]

        # enlarge raster geometry
        raster_geom_enlrgd = self.raster_geom.scale(2., inplace=False)
        extent_enlrgd = (self.extent[0] - extent_width/2.,
                         self.extent[1] - extent_height/2.,
                         self.extent[2] + extent_width/2.,
                         self.extent[3] + extent_height/2.)
        assert raster_geom_enlrgd.extent == extent_enlrgd

        # shrink raster geometry
        raster_geom_shrnkd = self.raster_geom.scale(0.5, inplace=False)
        extent_shrnkd = (self.extent[0] + extent_width / 4.,
                         self.extent[1] + extent_height / 4.,
                         self.extent[2] - extent_width / 4.,
                         self.extent[3] - extent_height / 4.)
        assert raster_geom_shrnkd.extent == extent_shrnkd

        # tests enlargement in pixels with a rotated geometry
        # TODO: width and height properties change for the rotated geometry -> discuss behaviour with others
        raster_geom_enlrgd = self.raster_geom_rot.scale(2., inplace=False)
        assert raster_geom_enlrgd.n_cols == (self.raster_geom_rot.n_cols * 2)
        assert raster_geom_enlrgd.n_rows == (self.raster_geom_rot.n_rows * 2)

    def test_resize(self):
        """ Tests resizing of a raster geometry by pixels and spatial reference units. """

        # tests resize in pixels
        buffer_sizes = [2, 3, 2, -3]
        raster_geom_reszd = self.raster_geom.resize(buffer_sizes, unit='px', inplace=False)
        assert raster_geom_reszd.n_rows == self.raster_geom.n_rows
        assert raster_geom_reszd.n_cols == (self.raster_geom.n_cols + 4)

        # tests resize in spatial reference units
        buffer_sizes = [self.x_pixel_size*2, self.y_pixel_size*3,
                        self.x_pixel_size*2, self.y_pixel_size*-3]
        raster_geom_reszd = self.raster_geom.resize(buffer_sizes, unit='sr', inplace=False)
        assert raster_geom_reszd.n_rows == self.raster_geom.n_rows
        assert raster_geom_reszd.n_cols == (self.raster_geom.n_cols + 4)

    def test_different_sref(self):
        """ Test topological operation if the spatial reference systems are different. """
        pass

    def test_equal(self):
        """ Tests if two raster geometries are equal (one created from an extent, one from a geometry). """

        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_geometry(self.geom, self.x_pixel_size, self.y_pixel_size, sref=self.sref)

        assert raster_geom_a == raster_geom_b

    def test_not_equal(self):
        """ Tests if two raster geometries are not equal. """

        # create raster geometries from the given extent
        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_extent(self.extent, self.sref, self.x_pixel_size, self.y_pixel_size)

        # resize second raster geometry
        raster_geom_b.scale(0.5, inplace=True)

        assert raster_geom_a != raster_geom_b

    def test_and(self):
        """ Tests AND operation, which is an intersection between both raster geometries. """

        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_geometry(self.geom, self.x_pixel_size, self.y_pixel_size, sref=self.sref)

        self.assertEqual(raster_geom_a & raster_geom_b, raster_geom_b & raster_geom_a)

    def test_in(self):
        """ Tests if another geometry is within a raster geometry. """

        # shrink raster geometry
        raster_geom_shrnkd = self.raster_geom.scale(0.5, inplace=False)
        assert raster_geom_shrnkd in self.raster_geom

        # enlarge raster geometry
        raster_geom_enlrgd = self.raster_geom.scale(2., inplace=False)
        assert raster_geom_enlrgd not in self.raster_geom

    def test_indexing(self):
        """ Tests `__get_item__` method, which is used for intersecting a raster geometry. """

        # test indexing within the boundaries of the raster geometry
        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        (ll_x, ll_y, ur_x, ur_y) = raster_geom_scaled.extent
        raster_geom_intsctd = self.raster_geom[ll_x:ur_x, ll_y:ur_y]

        assert raster_geom_scaled == raster_geom_intsctd

        # test indexing with new segment size
        raster_geom_intsctd = self.raster_geom[ll_x:ur_x:0.1, ll_y:ur_y:0.1]
        assert raster_geom_intsctd._segment_size == 0.1


# TODO: add randomness
class RasterGridTest(unittest.TestCase):
    """ Tests functionalities of `RasterGrid`. """

    def setUp(self):
        """
        Sets up a `RasterGrid` object.
        It is defined by 3x3 raster geometries/tiles in a LatLon projection.
        """

        x_pixel_size = 0.01
        y_pixel_size = -0.01
        grid_rows = 3
        grid_cols = 3
        grid_ul_x = 0.
        grid_ul_y = 60. + y_pixel_size  # ll pixel coordinate of ul grid coordinate
        rows = 1600
        cols = 1600

        sref = SpatialRef(4326)

        roi_x_min = grid_ul_x + cols/2.*x_pixel_size
        roi_y_max = grid_ul_y + rows/2.*y_pixel_size
        roi_x_max = roi_x_min + cols*x_pixel_size
        roi_y_min = roi_y_max + rows * y_pixel_size
        self.roi = [(roi_x_min, roi_y_min), (roi_x_max, roi_y_max)]

        raster_geoms = []
        for grid_row in range(grid_rows):
            for grid_col in range(grid_cols):
                ul_x = grid_ul_x + grid_col*cols*x_pixel_size
                ul_y = grid_ul_y + grid_row*rows*y_pixel_size
                gt = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
                tile_id = "S{:02d}W{:02d}".format(grid_row, grid_col)
                raster_geom = RasterGeometry(rows, cols, sref, geotrans=gt, geom_id=tile_id)
                raster_geoms.append(raster_geom)

        self.raster_grid = RasterGrid(raster_geoms)

    def test_tile_from_id(self):
        """ Tests retrieval of `RasterGeometry` tile by a given ID. """

        tile = self.raster_grid.tile_from_id("S01W01")
        assert tile.id == "S01W01"

    def test_neighbours(self):
        """  Tests retrieval of neighbouring tiles. """

        # tile situated in the center of the raster grid
        neighbours = self.raster_grid.neighbours_from_id("S01W01")
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = ["S00W00", "S00W01", "S00W02", "S01W00", "S01W02", "S02W00", "S02W01", "S02W02"]

        self.assertListEqual(neighbours_id, neighbours_id_should)

        # tile situated at the upper-right corner of the raster grid
        neighbours = self.raster_grid.neighbours_from_id("S00W02")
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = ["S00W01", "S01W01", "S01W02"]

        self.assertListEqual(neighbours_id, neighbours_id_should)

    def test_intersection_by_geom(self):
        """ Tests intersection of a geometry with the raster grid. """

        raster_grid_intsct = self.raster_grid.intersection_by_geom(self.roi, inplace=False)
        assert raster_grid_intsct.tile_ids == ["S00W00", "S00W01", "S01W00", "S01W01"]
        assert raster_grid_intsct.area == self.raster_grid["S00W00"].area

    def test_plotting(self):
        """ Tests plotting functionalities of a raster grid. """

        # most simple plot
        self.raster_grid.plot()

        # test plotting with labelling
        self.raster_grid.plot(proj=ccrs.PlateCarree(), label_tiles=True)

    def test_indexing(self):
        """ Tests `__get_item__` method, which is used for intersecting a raster grid or retrieving a tile. """

        # tile id indexing
        assert self.raster_grid["S01W01"].id == "S01W01"

        # spatial indexing
        extent_should = (self.roi[0][0], self.roi[0][1], self.roi[1][0], self.roi[1][1])
        raster_grid_intsct = self.raster_grid[extent_should[0]:extent_should[2], extent_should[1]:extent_should[3]]

        self.assertTupleEqual(raster_grid_intsct.outer_extent, extent_should)


if __name__ == '__main__':
    unittest.main()
