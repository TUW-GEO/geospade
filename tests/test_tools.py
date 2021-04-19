""" Test suite for the tools module. """

import unittest
import numpy as np
from shapely.geometry import Polygon
import ogr
from geospade.tools import get_quadrant
from geospade.tools import rasterise_polygon


class ToolsTest(unittest.TestCase):
    """ Tests all functions in the tools module. """

    def test_get_quadrant(self):
        """ Tests all 5 cases of quadrants (1, 2, 3, 4, None). """

        x = 1
        y = 1
        assert get_quadrant(x, y) == 1

        x = -1
        y = 1
        assert get_quadrant(x, y) == 2

        x = -1
        y = -1
        assert get_quadrant(x, y) == 3

        x = 1
        y = -1
        assert get_quadrant(x, y) == 4

        x = 1
        y = 0
        assert get_quadrant(x, y) is None

    def test_rasterise_polygon(self):
        """ Tests rasterisation of a polygon. """

        ref_raster = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
        ref_raster = np.array(ref_raster)
        poly_pts = [(1, 1), (1, 4), (5, 8), (6, 8), (6, 5), (8, 3), (6, 1), (1, 1)]
        geom = ogr.CreateGeometryFromWkt(Polygon(poly_pts).wkt)
        raster = rasterise_polygon(geom, 1, 1)

        assert np.all(raster == ref_raster)

        ref_raster = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 1, 0],
                               [1, 1, 1, 1, 0, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
        ref_raster = np.array(ref_raster)
        poly_pts = [(1, 1), (1, 7), (5, 3), (8, 6), (8, 1), (1, 1)]
        geom = ogr.CreateGeometryFromWkt(Polygon(poly_pts).wkt)
        raster = rasterise_polygon(geom, 1, 1)
        assert np.all(raster == ref_raster)

    def test_rasterise_polygon_buffer(self):
        """ Tests rasterisation of a polygon (with buffering). """

        ref_raster = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
        ref_raster = np.array(ref_raster)
        poly_pts = [(1, 1), (1, 4), (5, 8), (6, 8), (6, 5), (8, 3), (6, 1), (1, 1)]
        geom = ogr.CreateGeometryFromWkt(Polygon(poly_pts).wkt)
        raster = rasterise_polygon(geom, 1, 1, buffer=-1)

        assert np.all(raster == ref_raster)

        ref_raster = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 0]])
        ref_raster = np.array(ref_raster)
        poly_pts = [(1, 1), (1, 4), (5, 8), (6, 8), (6, 5), (8, 3), (6, 1), (1, 1)]
        geom = ogr.CreateGeometryFromWkt(Polygon(poly_pts).wkt)
        raster = rasterise_polygon(geom, 1, 1, buffer=1)

        assert np.all(raster == ref_raster)


if __name__ == '__main__':
    unittest.main()
