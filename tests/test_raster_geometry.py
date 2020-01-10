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

    def test_from_extent(self):
        """ Tests setting up a raster geometry from a given extent. """

        raster_geom = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)

        assert raster_geom.extent == self.extent

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
        raster_geom_a = RasterGeometry.from_extent(extent_a, self.sref, 0.5, -0.5)
        raster_geom_b = RasterGeometry.from_extent(extent_b, self.sref, 0.5, -0.5)
        raster_geom_c = RasterGeometry.from_extent(extent_c, self.sref, 0.5, -0.5)
        raster_geom_d = RasterGeometry.from_extent(extent_a, self.sref, 1., -1.)

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

        # create axis parallel raster geometry
        raster_geom_ap = RasterGeometry.from_geometry(self.geom, 0.5, -0.5, sref=self.sref)

        assert raster_geom_ap.is_axis_parallel

        # rotate geometry to be not axis parallel and create raster geometry from it
        geom_nap = affinity.rotate(self.geom, 45*np.pi/180., 'center')
        raster_geom_nap = RasterGeometry.from_geometry(geom_nap, 0.5, -0.5, sref=self.sref)

        assert not raster_geom_nap.is_axis_parallel

    def test_pixel_size(self):
        """ Check raster and vertical/horizontal pixel size. """

        # create axis parallel raster geometry
        raster_geom_ap = RasterGeometry.from_geometry(self.geom, 0.5, -0.5, sref=self.sref)

        # rotate geometry to be not axis parallel and create raster geometry from it
        geom_nap = affinity.rotate(self.geom, 45 * np.pi / 180., 'center')
        raster_geom_nap = RasterGeometry.from_geometry(geom_nap, 0.5, -0.5, sref=self.sref)

        assert raster_geom_ap.x_pixel_size == raster_geom_ap.h_pixel_size
        assert raster_geom_ap.y_pixel_size == raster_geom_ap.v_pixel_size
        assert raster_geom_nap.x_pixel_size != raster_geom_nap.h_pixel_size
        assert raster_geom_nap.y_pixel_size != raster_geom_nap.v_pixel_size

    def test_area(self):
        """ Checks computation of the area covered by the raster geometry. """

        # create raster geometry
        raster_geom = RasterGeometry.from_extent(self.extent, self.sref, 0.5, -0.5)
        # compute area of the extent
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]
        extent_area = extent_height*extent_width

        assert extent_area == raster_geom.area

    def test_vertices(self):
        """ Tests . """

    def test_intersection(self):
        """ Test intersection with different geometries. """


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

    def test_coord_conversion(self):
        r_0 = random.randint(0, self.rg_ext.rows)
        c_0 = random.randint(0, self.rg_ext.cols)

        x, y = self.rg_ext.rc2xy(r_0, c_0)
        r, c = self.rg_ext.xy2rc(x, y)

        self.assertTupleEqual((r_0, c_0), (r, c))

    def test_plot(self):
        pass

if __name__ == '__main__':
    unittest.main()
