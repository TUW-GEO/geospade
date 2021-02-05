========
geospade
========

*geospade* stands for **geosp**\ atial **a**\ bstract **d**\ efinition **e**\ nvironment.
It serves as place to define classes and properties of raster and vector geometries and their operations alike.
On a higher level, abstract definitions should be embedded in a geospatial context to support interaction with other Python packages, e.g. *gdal*, *geopandas* or *xarray*.
In comparison to these Python packages, *geospade* never touches or writes any geospatial data on disk.
It is a toolkit for geospatial entities (e.g., points, polygons, ...) and their relationships (e.g., intersection, within, ...) in a spatial reference system (e.g., reprojection, grids, ...).

In *geospade* a geospatial context is given by the spatial reference system class `SpatialRef`, which allows to convert between different spatial reference definitions in Python (e.g., *osr*, *cartopy.crs*, ...) and offers well-known spatial reference system representations, i.e., WKT, PROJ4, and EPSG.
It aims to solve discrepancies between these representations and to fix issues occurring in lower-level packages, e.g. *gdal*.

An abstract, geospatial definition of a raster is implemented in `RasterGeometry`.
It is constructed by providing a pixel extent (i.e., number of rows and columns), the 6 affine geotransformation parameters and a spatial reference system.
With this knowledge, one can use a `RasterGeometry` instance to do many operations, e.g. intersect it with geometries, transform between pixel and spatial reference system coordinates, resize it or interact with other raster geometries.

Often, geospatial image data is available in tiled or gridded format due to storage/memory limits.
To preserve the spatial relationship for each image, `MosaicGeometry` can help to apply geospatial operations across image boundaries.
It represents a simple collection of `RasterGeometry` instances, where each `RasterGeometry` describes the spatial properties of an image, i.o.w. a tile.
With this setup, tile relationships and neighbourhoods can be derived.


Contents
========

.. toctree::
   :maxdepth: 3

   Spatial Reference <notebooks/spatial_ref>
   Raster Geometry <notebooks/raster_geometry>
   Mosaic Geometry <notebooks/mosaic_geometry>
   License <license>
   Authors <authors>
   Changelog <changelog>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
