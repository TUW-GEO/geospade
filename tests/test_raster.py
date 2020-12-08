import ogr
import unittest
import random
import shapely
import shapely.wkt
import numpy as np
import cartopy.crs as ccrs
from shapely import affinity
from shapely.geometry import Polygon

from geospade.crs import SpatialRef
from geospade.raster import RasterGeometry
from geospade.raster import MosaicGeometry


class RasterGeometryTest(unittest.TestCase):
    """ Tests functionality of `RasterGeometry`. """

    def setUp(self):

        # set up spatial reference system
        self.sref = SpatialRef(4326)

        # define region of interest/extent
        ll_x = random.randrange(-50., 50., 10.)
        ll_y = random.randrange(-50., 50., 10.)
        ur_x = ll_x + random.randrange(10., 50., 10.)
        ur_y = ll_y + random.randrange(10., 50., 10.)
        self.sh_geom = Polygon((
            (ll_x, ll_y),
            (ll_x, ur_y),
            (ur_x, ur_y),
            (ur_x, ll_y)
        ))  # Polygon in clock-wise order
        self.ogr_geom = ogr.CreateGeometryFromWkt(self.sh_geom.wkt)
        self.ogr_geom.AssignSpatialReference(self.sref.osr_sref)

        self.extent = tuple(map(float, (ll_x, ll_y, ur_x, ur_y)))
        self.x_pixel_size = 0.5
        self.y_pixel_size = 0.5
        self.raster_geom = RasterGeometry.from_extent(self.extent, self.sref,
                                                      self.x_pixel_size, self.y_pixel_size)

        # create rotated raster geometry
        geom_nap = affinity.rotate(shapely.wkt.loads(self.ogr_geom.ExportToWkt()), 45, 'center')
        geom_nap = ogr.CreateGeometryFromWkt(geom_nap.wkt)
        geom_nap.AssignSpatialReference(self.sref.osr_sref)
        self.raster_geom_rot = RasterGeometry.from_geometry(geom_nap, self.x_pixel_size, self.y_pixel_size)

    def test_from_extent(self):
        """ Tests setting up a raster geometry from a given extent. """

        self.assertTupleEqual(self.raster_geom.outer_boundary_extent, self.extent)

    def test_from_geom(self):
        """ Tests setting up a raster geometry from a given geometry. """

        raster_geom = RasterGeometry.from_geometry(self.ogr_geom, self.x_pixel_size, self.y_pixel_size)

        self.assertListEqual(raster_geom.outer_boundary_corners, list(self.sh_geom.exterior.coords)[:-1])

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

        self.assertTupleEqual(raster_geom.outer_boundary_extent, new_extent)

        # test if error is raised, when raster geometries with different resolutions are joined
        try:
            raster_geom = RasterGeometry.from_raster_geometries([raster_geom_a, raster_geom_b, raster_geom_c,
                                                              raster_geom_d])
        except ValueError:
            assert True

    def test_parent(self):
        """ Tests accessing of parent `RasterData` objects. """

        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_scaled, inplace=False)

        assert raster_geom_intsct.parent == self.raster_geom

        # tests finding the parent root
        raster_geom_scaled = raster_geom_scaled.scale(0.5, inplace=False)
        raster_geom_intsct = raster_geom_intsct.slice_by_geom(raster_geom_scaled, inplace=False)

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

    def test_size(self):
        """ Checks computation of the size of the raster geometry. """

        # compute size of the extent
        extent_px_width = (self.extent[2] - self.extent[0])*int(1./self.x_pixel_size)
        extent_px_height = (self.extent[3] - self.extent[1])*int(1./self.y_pixel_size)

        extent_px_size = extent_px_height*extent_px_width

        assert extent_px_size == self.raster_geom.size

    def test_vertices(self):
        """ Tests if extent nodes are equal to the vertices of the raster geometry. """

        # create vertices from extent
        vertices = [(self.extent[0], self.extent[1]),
                    (self.extent[0], self.extent[3]),
                    (self.extent[2], self.extent[3]),
                    (self.extent[2], self.extent[1])]

        self.assertListEqual(self.raster_geom.outer_boundary_corners, vertices)

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
        raster_geom_intsct = self.raster_geom.slice_by_geom(self.raster_geom.boundary, inplace=False)
        assert self.raster_geom == raster_geom_intsct

        # test intersection with scaled geometry
        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_scaled, inplace=False)

        assert raster_geom_scaled == raster_geom_intsct

        # test intersection with partial overlap
        extent_shifted = tuple(np.array(self.extent) + 2)
        extent_intsct = (extent_shifted[0], extent_shifted[1], self.extent[2], self.extent[3])
        raster_geom_shifted = RasterGeometry.from_extent(extent_shifted, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_shifted, inplace=False)

        assert raster_geom_intsct.outer_boundary_extent == extent_intsct

        # test intersection with no overlap
        extent_no_ovlp = (self.extent[2] + 1., self.extent[3] + 1.,
                          self.extent[2] + 5., self.extent[3] + 5.)
        raster_geom_no_ovlp = RasterGeometry.from_extent(extent_no_ovlp, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_no_ovlp, inplace=False)

        assert raster_geom_intsct is None

    def test_intersection_snap_to_grid(self):
        """ Tests intersection with and without the `snap_to_grid` option. """

        # create raster geometry which has a slightly smaller extent
        raster_geom_reszd = self.raster_geom.resize(-abs(self.x_pixel_size)/5., unit='sr', inplace=False)

        # execute intersection with 'snap_to_grid=True'
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_reszd, snap_to_grid=True, inplace=False)
        assert raster_geom_intsct == self.raster_geom

        # execute intersection with 'snap_to_grid=False'
        raster_geom_intsct = self.raster_geom.slice_by_geom(raster_geom_reszd, snap_to_grid=False, inplace=False)
        assert raster_geom_intsct != self.raster_geom

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

    def test_coord_conversion(self):
        """ Tests bijective coordinate conversion. """

        r_0 = random.randint(0, self.raster_geom.n_rows)
        c_0 = random.randint(0, self.raster_geom.n_cols)

        x, y = self.raster_geom.rc2xy(r_0, c_0)
        r, c = self.raster_geom.xy2rc(x, y)

        self.assertTupleEqual((r_0, c_0), (r, c))

    def test_plot(self):
        """ Tests plotting function of a raster geometry. """

        self.raster_geom.plot(add_country_borders=True)

        # test plotting with labelling and different output projection
        self.raster_geom.id = "E048N018T1"
        self.raster_geom.plot(proj=ccrs.EckertI(), label_geom=True, add_country_borders=True)

        # test plotting with different input projection
        sref = SpatialRef(31259)
        extent = [527798, 94878, 956835, 535687]
        raster_geom = RasterGeometry.from_extent(extent, sref, x_pixel_size=500, y_pixel_size=500)
        raster_geom.plot(add_country_borders=True)

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
        assert raster_geom_enlrgd.outer_boundary_extent == extent_enlrgd

        # shrink raster geometry
        raster_geom_shrnkd = self.raster_geom.scale(0.5, inplace=False)
        extent_shrnkd = (self.extent[0] + extent_width / 4.,
                         self.extent[1] + extent_height / 4.,
                         self.extent[2] - extent_width / 4.,
                         self.extent[3] - extent_height / 4.)
        assert raster_geom_shrnkd.outer_boundary_extent == extent_shrnkd

        # tests enlargement in pixels with a rotated geometry
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
        # create raster geometry which touches the previous one
        extent_tchs = (self.extent[2], self.extent[3],
                       self.extent[2] + 5., self.extent[3] + 5.)
        raster_geom_tchs = RasterGeometry.from_extent(extent_tchs, self.sref,
                                                      self.x_pixel_size, self.y_pixel_size)
        # reproject to different system
        sref_other = SpatialRef(3857)
        geom = raster_geom_tchs.boundary_ogr
        geom.TransformTo(sref_other.osr_sref)
        assert self.raster_geom.touches(geom)

    def test_equal(self):
        """ Tests if two raster geometries are equal (one created from an extent, one from a geometry). """

        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_geometry(self.ogr_geom, self.x_pixel_size, self.y_pixel_size)

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
        raster_geom_b = RasterGeometry.from_geometry(self.ogr_geom, self.x_pixel_size, self.y_pixel_size)

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

        # test indexing with coordinates
        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        (ll_x, ll_y, ur_x, ur_y) = raster_geom_scaled.outer_boundary_extent
        outer_ur_x = ur_x + self.x_pixel_size
        outer_ur_y = ur_y + self.y_pixel_size
        raster_geom_intsctd = self.raster_geom[ll_x:outer_ur_x, ll_y:outer_ur_y, self.raster_geom.sref]
        assert raster_geom_scaled == raster_geom_intsctd

        # test indexing with pixel slicing
        max_row, min_col = self.raster_geom.xy2rc(ll_x, ll_y)
        min_row, max_col = self.raster_geom.xy2rc(ur_x, ur_y)
        outer_max_row = max_row + 1
        outer_max_col = max_col + 1
        raster_geom_intsctd = self.raster_geom[min_row:outer_max_row, min_col:outer_max_col]
        assert raster_geom_scaled == raster_geom_intsctd


# class MosaicGeoemtryTest(unittest.TestCase):
#     """ Tests functionality of `MosaicGeometry`. """
#
#     def setUp(self):
#         """
#         Sets up a `MosaicGeometry` object.
#
#         """
#         # define spatial reference
#         sref = SpatialRef(4326)
#         # define pixel spacing
#         x_pixel_size = 0.01
#         y_pixel_size = 0.01
#
#         # define width and height of each tile (randomly)
#         tile_height = random.randint(30, 3000)
#         tile_width = random.randint(30, 3000)
#         # define origin and number of tiles (randomly)
#         mosaic_ul_x = random.randrange(-50., 50., 10.)
#         mosaic_ul_y = random.randrange(-50., 50., 10.)
#         mosaic_rows = random.randint(2, 6)
#         mosaic_cols = random.randint(2, 6)
#
#         #self.roi = [(roi_x_min, roi_y_min), (roi_x_max, roi_y_max)]
#
#         raster_geoms = []
#         for mosaic_row in range(mosaic_rows):
#             for mosaic_col in range(mosaic_cols):
#                 ul_x = mosaic_ul_x + mosaic_col*tile_width*x_pixel_size
#                 ul_y = mosaic_ul_y - mosaic_row*tile_height*y_pixel_size
#                 gt = (ul_x, x_pixel_size, 0, ul_y, 0, -y_pixel_size)
#                 tile_id = "S{:02d}W{:02d}".format(mosaic_row, mosaic_col)
#                 raster_geom = RasterGeometry(tile_height, tile_width, sref, geotrans=gt, geom_id=tile_id)
#                 raster_geoms.append(raster_geom)
#
#         self.mosaic_geom = MosaicGeometry.from_list(raster_geoms)
#
#     def test_tile_from_id(self):
#         """ Tests retrieval of `RasterGeometry` tile by a given ID. """
#
#         tile = self.mosaic_geom.tile_from_id("S01W01")
#         assert tile.id == "S01W01"
#
#     def test_neighbours(self):
#         """  Tests retrieval of neighbouring tiles. """
#
#         # tile situated in the center of the mosaic geometry
#         neighbours = self.mosaic_geom.neighbours_from_id("S01W01")
#         neighbours_id = sorted([neighbour.id for neighbour in neighbours])
#         neighbours_id_should = []
#
#         self.assertListEqual(neighbours_id, neighbours_id_should)
#
#         # tile situated at the upper-right corner of the raster grid
#         neighbours = self.raster_grid.neighbours_from_id("S00W02")
#         neighbours_id = sorted([neighbour.id for neighbour in neighbours])
#         neighbours_id_should = ["S00W01", "S01W01", "S01W02"]
#
#         self.assertListEqual(neighbours_id, neighbours_id_should)
#
#     def test_intersection_by_geom(self):
#         """ Tests intersection of a geometry with the raster grid. """
#
#         raster_grid_intsct = self.raster_grid.slice_by_geom(self.roi, inplace=False)
#         assert raster_grid_intsct.tile_ids == ["S00W00", "S00W01", "S01W00", "S01W01"]
#         assert raster_grid_intsct.area == self.raster_grid["S00W00"].area
#
#     def test_plotting(self):
#         """ Tests plotting functionalities of a raster grid. """
#
#         # most simple plot
#         self.raster_grid.plot()
#
#         # test plotting with labelling
#         self.raster_grid.plot(proj=ccrs.PlateCarree(), label_tiles=True)
#
#     def test_indexing(self):
#         """ Tests `__get_item__` method, which is used for intersecting a raster grid or retrieving a tile. """
#
#         # tile id indexing
#         assert self.raster_grid["S01W01"].id == "S01W01"
#
#         # spatial indexing
#         extent_should = (self.roi[0][0], self.roi[0][1], self.roi[1][0], self.roi[1][1])
#         raster_grid_intsct = self.raster_grid[extent_should[0]:extent_should[2], extent_should[1]:extent_should[3]]
#
#         self.assertTupleEqual(raster_grid_intsct.outer_extent, extent_should)


if __name__ == '__main__':
    unittest.main()
