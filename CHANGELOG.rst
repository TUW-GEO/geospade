=========
Changelog
=========

Version 0.2.3
=============

- fixed rounding of world system coordinates

Version 0.2.2
=============

- fixed wrong encoding of mask arrays
- added geometry reprojection to mosaic methods
- added new `poi2tile()` method

Version 0.2.1
=============

- minor bug fixes
- typing

Version 0.2.0
=============

- corrected wrong pixel window definition in `slice_by_rc()`
- added support for GDAL 3

Version 0.1.1
=============

- replaced own `rasterise_polygon()` implementation by PIL's `ImageDraw` class

Version 0.1.0
=============

- added raster support with `RasterGeometry`, `Tile`, `MosaicGeometry` and `RegularMosaicGeometry`
- added crs support with `SpatialRef` class
- added a set of tools for geospatial operations, e.g. rasterising a polygon
- added higher-level transformation wrappers
