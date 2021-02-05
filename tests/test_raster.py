import os
import ogr
import json
import unittest
import random
import shapely
import shapely.wkt
import numpy as np
import cartopy.crs as ccrs
from shapely import affinity
from shapely.geometry import Polygon

from geospade.crs import SpatialRef
from geospade.tools import any_geom2ogr_geom
from geospade.raster import RasterGeometry
from geospade.raster import MosaicGeometry
from geospade.raster import RegularMosaicGeometry


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

    def test_from_json(self):
        """ Tests the creation of a raster geometry from a JSON file containing a raster geometry definition. """
        tmp_filepath = "raster_geom.json"
        self.raster_geom.to_json(tmp_filepath)

        raster_geom = RasterGeometry.from_json(tmp_filepath)
        os.remove(tmp_filepath)
        assert raster_geom == self.raster_geom

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
        raster_geom_intsctd = self.raster_geom[ll_x:ur_x, ll_y:ur_y, self.raster_geom.sref]
        assert raster_geom_scaled == raster_geom_intsctd

        # test indexing with pixel slicing
        max_row, min_col = self.raster_geom.xy2rc(ll_x, ll_y)
        min_row, max_col = self.raster_geom.xy2rc(ur_x, ur_y)
        raster_geom_intsctd = self.raster_geom[min_row:max_row, min_col:max_col]
        assert raster_geom_scaled == raster_geom_intsctd

    def test_to_json(self):
        """ Tests the creation of a JSON file containing a raster geometry definition. """
        tmp_filepath = "raster_geom.json"
        self.raster_geom.to_json(tmp_filepath)

        with open(tmp_filepath, 'r') as tmp_file:
            json_dict = json.load(tmp_file)

        os.remove(tmp_filepath)

        assert json_dict['id'] == self.raster_geom.id
        assert json_dict['number_of_rows'] == self.raster_geom.n_rows
        assert json_dict['number_of_columns'] == self.raster_geom.n_cols
        assert json_dict['spatial_reference'] == self.raster_geom.sref.to_proj4_dict()
        assert tuple(json_dict['geotransformation']) == self.raster_geom.geotrans
        assert json_dict['pixel_origin'] == self.raster_geom.px_origin
        assert json_dict['description'] == self.raster_geom.description


class MosaicGeometryTest(unittest.TestCase):
    """ Tests functionality of `MosaicGeometry`. """

    def setUp(self):
        """
        Sets up a `MosaicGeometry` object.

        """
        # define spatial reference
        sref = SpatialRef(4326)
        # define pixel spacing
        x_pixel_size = 0.01
        y_pixel_size = 0.01
        # define origin and number of tiles (randomly)
        mosaic_ul_x = random.randrange(-50., 50., 10.)
        mosaic_ul_y = random.randrange(-50., 50., 10.)
        mosaic_rows = 1
        x_tile_size = 1.
        y_tile_size = 1.
        # define cols
        upper_mosaic_cols = 5
        middle_mosaic_cols = 3
        lower_mosaic_cols = 4
        # define tile size
        upper_x_tile_size = x_tile_size
        middle_x_tile_size = x_tile_size * (upper_mosaic_cols/middle_mosaic_cols)
        lower_x_tile_size = x_tile_size * (upper_mosaic_cols/lower_mosaic_cols)
        # define geotrans
        upper_geotrans = (mosaic_ul_x, x_pixel_size, 0., mosaic_ul_y, 0., -y_pixel_size)
        middle_geotrans = (mosaic_ul_x, x_pixel_size, 0., mosaic_ul_y - y_tile_size, 0., -y_pixel_size)
        lower_geotrans = (mosaic_ul_x, x_pixel_size, 0., mosaic_ul_y - 2*y_tile_size, 0., -y_pixel_size)

        upper_mosaic_geom = RegularMosaicGeometry.from_rectangular_definition(mosaic_rows, upper_mosaic_cols,
                                                                              upper_x_tile_size, y_tile_size,
                                                                              sref, geotrans=upper_geotrans)
        middle_mosaic_geom = RegularMosaicGeometry.from_rectangular_definition(mosaic_rows, middle_mosaic_cols,
                                                                               middle_x_tile_size, y_tile_size,
                                                                               sref, geotrans=middle_geotrans)
        lower_mosaic_geom = RegularMosaicGeometry.from_rectangular_definition(mosaic_rows, lower_mosaic_cols,
                                                                              lower_x_tile_size, y_tile_size,
                                                                              sref, geotrans=lower_geotrans)

        # merge tiles
        tiles = upper_mosaic_geom.tiles + middle_mosaic_geom.tiles + lower_mosaic_geom.tiles

        # relabel tile IDs
        for i, tile in enumerate(tiles):
            tile.id = i + 1

        self.mosaic_geom = MosaicGeometry(tiles)

    def test_tile_from_id(self):
        """ Tests retrieval of a tile in a `IrregularMosaicGeometry` instance by a given ID. """

        tile = self.mosaic_geom.id2tile(1)
        assert tile.id == 1

    def test_neighbours(self):
        """  Tests retrieval of neighbouring tiles. """

        # tile situated in the center of the mosaic geometry
        neighbours = self.mosaic_geom.get_neighbouring_tiles(7)
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = sorted([2, 3, 4, 6, 8, 10, 11])

        self.assertListEqual(neighbours_id, neighbours_id_should)

        # tile situated at the upper-right corner of the mosaic geometry
        neighbours = self.mosaic_geom.get_neighbouring_tiles(5)
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = sorted([4, 8])

        self.assertListEqual(neighbours_id, neighbours_id_should)

    def test_intersection_by_geom(self):
        """ Tests intersection of a geometry with the irregular mosaic geometry. """

        geom = any_geom2ogr_geom(self._get_roi(), sref=self.mosaic_geom.sref)
        mosaic_intsct = self.mosaic_geom.slice_by_geom(geom)
        assert len(mosaic_intsct.tile_ids) == 5

    def test_plotting(self):
        """ Tests plotting functionality of an irregular mosaic geometry """

        # most simple plot
        self.mosaic_geom.plot(show=False)

        # test plotting with labelling
        self.mosaic_geom.plot(proj=ccrs.EckertI(), label_tiles=True, show=False)

    def test_indexing(self):
        """ Tests `__get_item__` method, which is used for intersecting a regular mosaic or retrieving a tile. """

        # tile id indexing
        assert self.mosaic_geom[7].id == 7

        # spatial indexing
        roi = self._get_roi()
        outer_ur_x = roi[2] + self.mosaic_geom.x_pixel_size
        outer_ur_y = roi[3] + self.mosaic_geom.y_pixel_size
        mosaic_intsct = self.mosaic_geom[roi[0]:outer_ur_x, roi[1]:outer_ur_y, self.mosaic_geom.sref]
        outer_boundary_extent = RasterGeometry.from_raster_geometries(mosaic_intsct.tiles).outer_boundary_extent
        self.assertTupleEqual(outer_boundary_extent, roi)

    def test_to_from_json(self):
        """ Tests dumping a mosaic geometry to and loading it from disk. """
        tmp_filename = "mosaic_geom.json"
        self.mosaic_geom.to_json(tmp_filename)

        mosaic_geom = MosaicGeometry.from_json(tmp_filename)

        os.remove(tmp_filename)

        assert mosaic_geom.id == self.mosaic_geom.id
        assert mosaic_geom.description == self.mosaic_geom.description
        assert mosaic_geom.tile_ids == self.mosaic_geom.tile_ids
        assert np.all(mosaic_geom._adjacency_matrix == self.mosaic_geom._adjacency_matrix)

    def _get_roi(self):
        """ Helper function for retrieving the region of interest. """
        ll_tile = self.mosaic_geom[6]
        ur_tile = self.mosaic_geom[3]
        roi = (ll_tile.centre[0], ll_tile.centre[1], ur_tile.centre[0], ur_tile.centre[1])

        return roi


class RegularMosaicGeometryTest(unittest.TestCase):
    """ Tests functionality of `RegularMosaicGeometry`. """

    def setUp(self):
        """
        Sets up a `RegularMosaicGeometry` object.

        """
        # define spatial reference
        sref = SpatialRef(4326)
        # define pixel spacing
        x_pixel_size = 0.01
        y_pixel_size = 0.01
        # define origin and number of tiles (randomly)
        mosaic_ul_x = random.randrange(-50., 50., 10.)
        mosaic_ul_y = random.randrange(-50., 50., 10.)
        mosaic_rows = 3
        mosaic_cols = 3
        x_tile_size = 1.
        y_tile_size = 1.
        # define geotrans
        geotrans = (mosaic_ul_x, x_pixel_size, 0., mosaic_ul_y, 0., -y_pixel_size)

        self.mosaic_geom = RegularMosaicGeometry.from_rectangular_definition(mosaic_rows, mosaic_cols, x_tile_size,
                                                                             y_tile_size, sref, geotrans=geotrans)

    def test_tile_from_id(self):
        """ Tests retrieval of a tile in a `RegularMosaicGeometry` instance by a given ID. """

        tile = self.mosaic_geom.id2tile("S001W001")
        assert tile.id == "S001W001"

    def test_neighbours(self):
        """  Tests retrieval of neighbouring tiles. """

        # tile situated in the center of the mosaic geometry
        neighbours = self.mosaic_geom.get_neighbouring_tiles("S001W001")
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = sorted(["S000W000", "S001W000", "S002W000", "S000W001",
                                       "S002W001", "S000W002", "S001W002", "S002W002"])

        self.assertListEqual(neighbours_id, neighbours_id_should)

        # tile situated at the upper-right corner of the mosaic geometry
        neighbours = self.mosaic_geom.get_neighbouring_tiles("S000W002")
        neighbours_id = sorted([neighbour.id for neighbour in neighbours])
        neighbours_id_should = sorted(["S000W001", "S001W001", "S001W002"])

        self.assertListEqual(neighbours_id, neighbours_id_should)

    def test_intersection_by_geom(self):
        """ Tests intersection of a geometry with the regular mosaic geometry. """

        geom = any_geom2ogr_geom(self._get_roi(), sref=self.mosaic_geom.sref)
        mosaic_intsct = self.mosaic_geom.slice_by_geom(geom)
        assert len(mosaic_intsct.tile_ids) == 4

    def test_plotting(self):
        """ Tests plotting functionality of a regular mosaic geometry """

        # most simple plot
        self.mosaic_geom.plot(show=False)

        # test plotting with labelling
        self.mosaic_geom.plot(proj=ccrs.EckertI(), label_tiles=True, show=False)

    def test_indexing(self):
        """ Tests `__get_item__` method, which is used for intersecting a regular mosaic or retrieving a tile. """

        # tile id indexing
        assert self.mosaic_geom["S001W001"].id == "S001W001"

        # spatial indexing
        roi = self._get_roi()
        outer_ur_x = roi[2] + self.mosaic_geom.x_pixel_size
        outer_ur_y = roi[3] + self.mosaic_geom.y_pixel_size
        mosaic_intsct = self.mosaic_geom[roi[0]:outer_ur_x, roi[1]:outer_ur_y, self.mosaic_geom.sref]
        outer_boundary_extent = RasterGeometry.from_raster_geometries(mosaic_intsct.tiles).outer_boundary_extent
        self.assertTupleEqual(outer_boundary_extent, roi)

    def _get_roi(self):
        """ Helper function for retrieving the region of interest. """
        ll_tile = self.mosaic_geom["S001W000"]
        ur_tile = self.mosaic_geom["S000W001"]
        roi = (ll_tile.centre[0], ll_tile.centre[1], ur_tile.centre[0], ur_tile.centre[1])

        return roi


if __name__ == '__main__':
    unittest.main()
