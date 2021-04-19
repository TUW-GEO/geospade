""" Testing suite for the crs module. """

import osr
import unittest
from geospade.crs import SpatialRef
import cartopy.crs as ccrs


class TestSpatialref(unittest.TestCase):
    """ Test class for evaluating all functions of `SpatialRef`. """

    def setUp(self):
        """ Initialises static class variables for verification. """

        self.epsg = 31259
        self.wkt = 'PROJCS[\"MGI / Austria GK M34\",GEOGCS[\"MGI\",DATUM[\"Militar_Geographische_Institute\",SPHEROID' \
                   '[\"Bessel 1841\",6377397.155,299.1528128,AUTHORITY[\"EPSG\",\"7004\"]],' \
                   'TOWGS84[577.326,90.129,463.919,5.137,1.474,5.297,2.4232],AUTHORITY[\"EPSG\",\"6312\"]],' \
                   'PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,' \
                   'AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4312\"]],PROJECTION[\"Transverse_Mercator\"],' \
                   'PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",16.33333333333333],' \
                   'PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",750000],' \
                   'PARAMETER[\"false_northing\",-5000000],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],' \
                   'AUTHORITY[\"EPSG\",\"31259\"]]'
        self.proj4 = "+proj=tmerc +lat_0=0 +lon_0=16.33333333333333 +k=1 +x_0=750000 +y_0=-5000000 +ellps=bessel " \
                     "+towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232 +units=m +no_defs"
        self.proj4_dict = {'proj': 'tmerc', 'lat_0': 0.0, 'lon_0': 16.33333333333333, 'k': 1.0, 'x_0': 750000.0,
                           'y_0': -5000000.0, 'ellps': 'bessel',
                           'towgs84': '577.326,90.129,463.919,5.137,1.474,5.297,2.4232', 'units': 'm', 'no_defs': True}

    def test_create_from_epsg(self):
        """ Creates a `SpatialRef` object from an EPSG code and checks all conversions. """

        sref = SpatialRef(self.epsg)
        self.assertListEqual([self.epsg, self.wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

        sref = SpatialRef(self.epsg, sref_type="epsg")
        self.assertListEqual([self.epsg, self.wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

    def test_create_from_wkt(self):
        """ Creates a `SpatialRef` object from a WKT string and checks all conversions."""

        sref = SpatialRef(self.wkt)
        self.assertListEqual([self.epsg, self.wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

        sref = SpatialRef(self.wkt, sref_type="wkt")
        self.assertListEqual([self.epsg, self.wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

    def test_create_from_proj4(self):
        """ Creates a `SpatialRef` object from a PROJ4 string and checks all conversions."""

        # special wkt string has to be defined because PROJ4 -> WKT does lead to an unnamed projection
        wkt = 'PROJCS["unnamed",GEOGCS["Bessel 1841",DATUM["unknown",SPHEROID["bessel",6377397.155,299.1528128],' \
              'TOWGS84[577.326,90.129,463.919,5.137,1.474,5.297,2.4232]],PRIMEM["Greenwich",0],' \
              'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],' \
              'PARAMETER["central_meridian",16.33333333333333],PARAMETER["scale_factor",1],' \
              'PARAMETER["false_easting",750000],PARAMETER["false_northing",-5000000],UNIT["Meter",1]]'
        epsg = None  # None because EPSG <-> PROJ4 transformation is not possible/is ambiguous

        sref = SpatialRef(self.proj4)
        self.assertListEqual([epsg, wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

        sref = SpatialRef(self.proj4, sref_type="proj4")
        self.assertListEqual([epsg, wkt, self.proj4],
                             [sref.epsg, sref.wkt, sref.proj4])

    def test_proj4_dict(self):
        """ Tests the export of a PROJ4 string to a dictionary representation. """

        sref = SpatialRef(self.proj4, sref_type="proj4")
        assert sref.to_proj4_dict() == self.proj4_dict

    def test_from_osr(self):
        """ Tests `SpatialRef` class creation from an OSR spatial reference object. """

        # create OSR spatial reference from PROJ4 string
        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromProj4(self.proj4)

        # create SpatialRef object from OSR spatial reference
        sref = SpatialRef.from_osr(osr_sref)

        assert sref.proj4 == self.proj4

    def test_to_pretty_wkt(self):
        """ Tests pretty Well Known Text (WKT) export. """

        pretty_wkt = 'PROJCS["MGI / Austria GK M34",\n    GEOGCS["MGI",\n        ' \
                     'DATUM["Militar_Geographische_Institute",\n            ' \
                     'SPHEROID["Bessel 1841",6377397.155,299.1528128,\n                ' \
                     'AUTHORITY["EPSG","7004"]],\n            ' \
                     'TOWGS84[577.326,90.129,463.919,5.137,1.474,5.297,2.4232],\n            ' \
                     'AUTHORITY["EPSG","6312"]],\n        PRIMEM["Greenwich",0,\n            ' \
                     'AUTHORITY["EPSG","8901"]],\n        UNIT["degree",0.0174532925199433,\n            ' \
                     'AUTHORITY["EPSG","9122"]],\n        AUTHORITY["EPSG","4312"]],\n    ' \
                     'PROJECTION["Transverse_Mercator"],\n    PARAMETER["latitude_of_origin",0],\n    ' \
                     'PARAMETER["central_meridian",16.33333333333333],\n    PARAMETER["scale_factor",1],\n    ' \
                     'PARAMETER["false_easting",750000],\n    PARAMETER["false_northing",-5000000],\n    ' \
                     'UNIT["metre",1,\n        AUTHORITY["EPSG","9001"]],\n    AUTHORITY["EPSG","31259"]]'

        sref = SpatialRef(self.epsg)
        assert sref.to_pretty_wkt() == pretty_wkt

    def test_to_cartopy_proj(self):
        """  Tests Cartopy projection creation from a `SpatialRef` instance. """
        sref = SpatialRef(4326)
        capy_proj = sref.to_cartopy_proj()
        assert isinstance(capy_proj, ccrs.PlateCarree)

        sref = SpatialRef(self.epsg)
        capy_proj = sref.to_cartopy_proj()
        assert isinstance(capy_proj, ccrs.TransverseMercator)

        sref = SpatialRef(102018)
        capy_proj = sref.to_cartopy_proj()
        assert isinstance(capy_proj, ccrs.Stereographic)

    def test_equal(self):
        """ Tests if two `SpatialRef` instances are equal. """

        sref_a = SpatialRef(self.proj4, sref_type='proj4')
        sref_b = SpatialRef(self.proj4_dict, sref_type='proj4')

        assert sref_a == sref_b

    def test_not_equal(self):
        """ Tests if two `SpatialRef` instances are not equal. """

        sref_a = SpatialRef(self.proj4, sref_type='proj4')
        sref_b = SpatialRef(4326, sref_type='epsg')

        assert sref_a != sref_b


if __name__ == '__main__':
    unittest.main()
