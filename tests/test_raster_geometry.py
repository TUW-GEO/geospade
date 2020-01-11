import unittest
import random
import shapely
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from geospade.spatial_ref import SpatialRef
from geospade.definition import RasterGeometry


class TestRasterGeometry(unittest.TestCase):
    """ Tests functionalities of `RasterGeometry`. """

    def setUp(self):

        # set up spatial reference system
        self.sref = SpatialRef(4326)

        # define region of interest/extent
        ll_x = random.randrange(-50, 50, 10)
        ll_y = random.randrange(-50, 50, 10)
        ur_x = ll_x + random.randrange(10, 50, 10)
        ur_y = ll_y + random.randrange(10, 50, 10)
        self.geom = Polygon((
            (ll_x, ll_y),
            (ur_x, ll_y),
            (ur_x, ur_y),
            (ll_x, ur_y)
        ))
        self.extent = (ll_x, ll_y, ur_x, ur_y)
        self.x_pixel_size = 0.5
        self.y_pixel_size = -0.5
        self.raster_geom = RasterGeometry.from_extent(self.extent, self.sref, self.x_pixel_size, self.y_pixel_size)

        # create rotated raster geometry
        geom_nap = affinity.rotate(self.geom, 45 * np.pi / 180., 'center')
        self.raster_geom_rot = RasterGeometry.from_geometry(geom_nap, self.x_pixel_size, self.y_pixel_size,
                                                            sref=self.sref)

    def test_from_extent(self):
        """ Tests setting up a raster geometry from a given extent. """

        assert self.raster_geom.extent == self.extent

    def test_from_geom(self):
        """ Tests setting up a raster geometry from a given geometry. """

        raster_geom = RasterGeometry.from_geometry(self.geom, 0.5, -0.5, sref=self.sref)

        assert shapely.loads.wkt(raster_geom.boundary.ExportToWkt()) == self.geom

    def test_get_common_geom(self):
        """ Tests the creation of an encasing raster geometry from multiple raster geometries. """

        # create different raster geometries
        extent_a = self.extent
        # compute hypotenuse of extent as a shift parameter
        extent_width = extent_a[2] - extent_a[0]
        extent_height = extent_a[3] - extent_a[1]
        extent_hypo = np.hypot(extent_width, extent_height)
        extent_b = tuple(np.array(extent_a) - extent_hypo)  # shift extent negatively
        extent_c = tuple(np.array(extent_a) + extent_hypo)  # shift extent positively
        # create raster geometries from different extents
        raster_geom_a = self.raster_geom
        raster_geom_b = RasterGeometry.from_extent(extent_b, self.sref, self.x_pixel_size, self.y_pixel_size)
        raster_geom_c = RasterGeometry.from_extent(extent_c, self.sref, self.x_pixel_size, self.y_pixel_size)
        raster_geom_d = RasterGeometry.from_extent(extent_a, self.sref, self.x_pixel_size*2, self.y_pixel_size*2)

        # resize the main geometry
        raster_geom_a.scale(0.5)

        # create common raster geometry from the first three raster geometries
        raster_geom = RasterGeometry.get_common_geometry([raster_geom_a, raster_geom_b, raster_geom_c])
        new_extent = (raster_geom_b.ll_x, raster_geom_b.ll_y, raster_geom_c.ur_x, raster_geom_c.ur_y)

        assert raster_geom.extent == new_extent

        # test if error is raised, when raster geometries with different resolutions are joined
        try:
            raster_geom = RasterGeometry.get_common_geometry([raster_geom_a, raster_geom_b, raster_geom_c,
                                                              raster_geom_d])
        except ValueError:
            assert True

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

        assert self.raster_geom.vertices == vertices

    def test_intersection(self):
        """ Test intersection with different geometries. """

        # test intersection with own geometry
        raster_geom_intsct = self.raster_geom.intersection(self.raster_geom.boundary, inplace=False)
        assert self.raster_geom == raster_geom_intsct

        # test intersection with scaled geometry
        raster_geom_scaled = self.raster_geom.scale(0.5, inplace=False)
        raster_geom_intsct = self.raster_geom.intersection(raster_geom_scaled)

        assert raster_geom_scaled == raster_geom_intsct

        # test intersection with partial overlap
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]
        ll_x_shifted = self.extent[0] + extent_width/2.
        ll_y_shifted = self.extent[1] + extent_height/2.
        ur_x_shifted = ll_x_shifted + extent_width
        ur_y_shifted = ll_y_shifted + extent_height
        extent_shifted = (ll_y_shifted, ll_x_shifted, ur_x_shifted, ur_y_shifted)
        extent_intsct = (ll_y_shifted, ll_x_shifted, self.extent[2], self.extent[3])
        raster_geom_shifted = RasterGeometry.from_extent(extent_shifted, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.intersection(raster_geom_shifted, inplace=False)

        assert raster_geom_intsct.extent == extent_intsct

        # test intersection with no overlap
        extent_no_ovlp = (self.extent[2] + 1., self.extent[3] + 1.,
                          self.extent[2] + extent_width/2., self.extent[3] + extent_height/2.)
        raster_geom_no_ovlp = RasterGeometry.from_extent(extent_no_ovlp, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        raster_geom_intsct = self.raster_geom.intersection(raster_geom_no_ovlp, inplace=False)

        assert raster_geom_intsct is None

    def test_intersection_snap_to_grid(self):
        """ Tests intersection with and without the `snap_to_grid` option. """

        # create raster geometry which has a slightly smaller extent
        raster_geom_reszd = self.raster_geom.resize(-abs(self.x_pixel_size)/5., unit='sr', inplace=False)
        
        # execute intersection with 'snap_to_grid=True'
        raster_geom_intsct = self.raster_geom.intersection(raster_geom_reszd, snap_to_grid=True, inplace=False)
        assert raster_geom_intsct == self.raster_geom

        # execute intersection with 'snap_to_grid=False'
        raster_geom_intsct = self.raster_geom.intersection(raster_geom_reszd, snap_to_grid=False, inplace=False)
        assert raster_geom_intsct != self.raster_geom

    def test_segment_size(self):
        """ Tests intersections with different segment sizes. """
        pass

    def test_touches(self):
        """ Tests if raster geometries touch each other. """

        # create raster geometry which touches the previous one
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]
        extent_tchs = (self.extent[2], self.extent[3],
                       self.extent[2] + extent_width/2., self.extent[3] + extent_height/2.)
        raster_geom_tchs = RasterGeometry.from_extent(extent_tchs, self.sref,
                                                      self.x_pixel_size, self.y_pixel_size)
        assert self.raster_geom.touches(raster_geom_tchs)

        # create raster geometry which does not touch the previous one
        extent_no_tchs = (self.extent[2] + 1., self.extent[3] + 1.,
                       self.extent[2] + extent_width / 2., self.extent[3] + extent_height / 2.)
        raster_geom_no_tchs = RasterGeometry.from_extent(extent_no_tchs, self.sref,
                                                         self.x_pixel_size, self.y_pixel_size)
        assert not self.raster_geom.touches(raster_geom_no_tchs)

    def test_to_cartopy_crs(self):
        """ Tests creation of a Cartopy CRS from a raster geometry. """
        pass

    def test_coord_conversion(self):
        """ Tests bijective coordinate conversion. """

        r_0 = random.randint(0, self.raster_geom.rows)
        c_0 = random.randint(0, self.raster_geom.cols)

        x, y = self.raster_geom.rc2xy(r_0, c_0)
        r, c = self.raster_geom.xy2rc(x, y)

        self.assertTupleEqual((r_0, c_0), (r, c))

    def test_plot(self):
        """ Tests plotting function of a raster geometry. """

        self.raster_geom.plot(proj=self.raster_geom.to_cartopy_crs())

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
        extent_shrnkd = (self.extent[0] - extent_width / 4.,
                         self.extent[1] - extent_height / 4.,
                         self.extent[2] + extent_width / 4.,
                         self.extent[3] + extent_height / 4.)
        assert raster_geom_shrnkd.extent == extent_shrnkd

        # tests enlargement in pixels with a rotated geometry
        raster_geom_enlrgd = self.raster_geom_rot.scale(2., inplace=False)
        assert raster_geom_enlrgd.rows == self.raster_geom_rot.rows * 2
        assert raster_geom_enlrgd.cols == self.raster_geom_rot.cols * 2

    def test_resize(self):
        """ Tests resizing of a raster geometry by pixels and spatial reference units. """

        # tests resize in pixels
        raster_geom_reszd = self.raster_geom.resize([2, 3, 2, -3], unit='px', inplace=False)
        assert raster_geom_reszd.rows == (self.raster_geom.rows + 4)
        assert raster_geom_reszd.cols == self.raster_geom.cols

        # tests resize in spatial reference units
        raster_geom_reszd = self.raster_geom.resize([self.x_pixel_size*2, self.y_pixel_size*3,
                                                     self.x_pixel_size*2, self.y_pixel_size*-3],
                                                    unit='sr', inplace=False)
        assert raster_geom_reszd.rows == (self.raster_geom.rows + 4)
        assert raster_geom_reszd.cols == self.raster_geom.cols

    def test_equal(self):
        """ Tests if two raster geometries are equal (one created from an extent, one from a geometry). """

        raster_geom_a = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)
        raster_geom_b = RasterGeometry.from_geometry(self.geom, 0.5, -0.5, sref=self.sref)

        assert raster_geom_a == raster_geom_b

    def test_not_equal(self):
        """ Tests if two raster geometries are not equal. """

        # create raster geometries from the given extent
        raster_geom_a = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)
        raster_geom_b = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)

        # resize second raster geometry
        raster_geom_b.scale(0.5)

        assert raster_geom_a != raster_geom_b

    def test_and(self):
        """ Tests AND operation, which is an intersection between both raster geometries. """

        raster_geom_a = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)
        raster_geom_b = RasterGeometry.from_geometry(self.geom, 0.5, -0.5, sref=self.sref)

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



        # test indexing with new segment size

if __name__ == '__main__':
    unittest.main()
