=========
Changelog
=========

Version 0.1.1
=============

- replaced own `rasterise_polygon()` implementation by PIL's `ImageDraw` class

Version 0.1.0
=============

- added raster support with `RasterGeometry`, `Tile`, `MosaicGeometry` and `RegularMosaicGeometry`
- added crs support with `SpatialRef` class
- added a set of tools for geospatial operations, e.g. rasterising a polygon
- added higher-level transformation wrappers
