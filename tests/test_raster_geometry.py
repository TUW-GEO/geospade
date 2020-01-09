import unittest
import random
from shapely.geometry import Polygon
from geospade.spatial_ref import SpatialRef
from geospade.definition import RasterGeometry


class TestRasterGeometry(unittest.TestCase):

    def setUp(self):
        self.sref = SpatialRef(4326)

        ll_x = random.randrange(-50, 50, 10)
        ll_y = random.randrange(-50, 50, 10)
        ur_x = ll_x + random.randrange(10, 50, 10)
        ur_y = ll_y + random.randrange(10, 50, 10)
        pgon = Polygon((
            (ll_x, ll_y),
            (ur_x, ll_y),
            (ur_x, ur_y),
            (ll_x, ur_y)
        ))
        self.rg_ext = RasterGeometry.from_extent((ll_x, ll_y, ur_x, ur_y),
                                                 self.sref,
                                                 0.5,
                                                 -0.5)

        self.rg_geom = RasterGeometry.from_geometry(pgon, 0.5, -0.5, sref=self.sref)

    def test_create(self):
        self.assertEqual(self.rg_ext, self.rg_geom)

    def test_intersect(self):
        self.assertEqual(self.rg_ext & self.rg_geom, self.rg_geom & self.rg_ext)

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
