<img align="right" src="https://github.com/TUW-GEO/geospade/raw/master/docs/imgs/geospade_logo.png" height="400" width="400">

# geospade
[![Build Status](https://travis-ci.com/TUW-GEO/geospade.svg?branch=master)](https://travis-ci.org/TUW-GEO/geospade)
[![Coverage Status](https://coveralls.io/repos/github/TUW-GEO/geospade/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/geospade?branch=master)
[![PyPi Package](https://badge.fury.io/py/geospade.svg)](https://badge.fury.io/py/geospade)
[![RTD](https://readthedocs.org/projects/geospade/badge/?version=latest)](https://geospade.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
*geospade* stands for **geosp**atial **a**bstract **d**efinition **e**nvironment. 
It serves as place to define classes and properties of raster and vector geometries and their operations alike.
On a higher level, abstract definitions should be embedded in a geospatial context to support interaction with other Python packages, e.g. *gdal*, *geopandas* or *xarray*.
In comparison to these Python packages, *geospade* never touches or writes any geospatial data on disk. 
It is a toolkit for geospatial entities (e.g., points, polygons, ...) and their relations (e.g., intersection, within, ...) in a spatial reference system (e.g., reprojection, mosaics, ...). 

In *geospade* a geospatial context is given by the spatial reference system class `crs.SpatialRef`, which allows to convert between different spatial reference definitions in Python (e.g., *osr*, *cartopy.crs*, ...) and offers well-known spatial reference system representations, i.e. WKT, PROJ4, and EPSG.
It aims to solve discrepancies between these representations and lower-level package versions, e.g. *gdal*.

An abstract, geospatial definition of a raster is implemented in `RasterGeometry`. 
It is constructed by providing a pixel extent, i.e. the number of rows and columns, the 6 affine geotransformation parameters and a spatial reference system.
With this knowledge, one can use a `RasterGeometry` instance to do many operations, e.g. intersect it with geometries, transform between pixel and spatial reference system coordinates, resize it, or interact with other raster geometries.

Often, geospatial image data is available in tiled or gridded format due to storage/memory limits. 
To preserve the spatial relationship for each image, `MosaicGeometry` can help to apply geospatial operations across image/tile boundaries.
It represents a simple collection of `Tile`/`RasterGeometry` instances, where each `Tile` describes the spatial properties of an image.
With this setup, tile relations and neighbourhoods can be derived.

## Installation
The package can be either installed via pip or if you solely want to work with *geospade* or contribute, we recommend to 
install it as a conda environment. If you work already with your own environment, please have look at ``conda_environment.yml`` and install/adapt missing packages.

### pip
To install *geospade* via pip in your own environment, use:
```
pip install geospade
```

### conda
The packages also comes along with a conda environment ``conda_environment.yml``. 
This is especially recommended if you want to contribute to the project.
The following script will install miniconda and setup the environment on a UNIX
like system. Miniconda will be installed into ``$HOME/miniconda``.
```
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda env create -f conda_env.yml
source activate geospade
```
This script adds ``$HOME/miniconda/bin`` temporarily to the ``PATH`` to do this
permanently add ``export PATH="$HOME/miniconda/bin:$PATH"`` to your ``.bashrc``
or ``.zshrc``.

For Windows, use the following setup:
  * Download the latest [miniconda 3 installer](https://docs.conda.io/en/latest/miniconda.html) for Windows
  * Click on ``.exe`` file and complete the installation.
  * Add the folder ``condabin`` folder to your environment variable ``PATH``. 
    You can find the ``condabin`` folder usually under: ``C:\Users\username\AppData\Local\Continuum\miniconda3\condabin``
  * Finally, you can set up the conda environment via:
    ```
    conda env create -f conda_environment.yml
    source activate geospade
    ```
    
After that you should be able to run 
```
python setup.py test
```
to run the test suite or 
```
python setup.py install
```
to install *geospade*.

## Contribution
We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.
If you want to contribute please follow these steps:

  * Fork the *geospade* repository to your account
  * Clone the *geospade* repository
  * Make a new feature branch from the *geospade* master branch
  * Add your feature
  * Please include tests for your contributions in one of the test directories.
    We use *py.test* so a simple function called ``test_my_feature`` is enough
  * Submit a pull request to our master branch