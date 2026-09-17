Installation
=============

Prerequisites
--------------

Install `poetry <https://python-poetry.org/docs/#installation>`_. Theia
requires Python 3.12 or later.

Installing the package
------------------------

.. code-block:: bash

   poetry install

Terrain data
-------------

Theia uses `SRTM <https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1.003>`_
elevation tiles (``.hgt`` files) to account for terrain occlusion and earth
curvature. Place the tiles you need for your scenario's region in the
directory pointed to by the ``THEIA_DEFAULT_TERRAIN`` configuration.

.. todo::

   Document the source we use internally to obtain SRTM tiles, and link it
   here.

For faster repeated line-of-sight queries over a fixed region, a
precomputed terrain tree can be built with ``scripts/create_terrain.py``.
See :mod:`theia.terrain_fast_los` for details.

Example scenarios and sensor configurations
---------------------------------------------

The ``scenarios/`` directory contains example scenario files (e.g. ballistic
missile defence, drone defence, static and mobile deployments) that can be
used as a starting point.

The ``data/default_configurations/`` directory contains default sensor
configurations (radar, SHORAD, surveillance). These are parametrised after
public specifications of real systems where possible — for example, the
default active radar configuration is derived from the Hensoldt TRML-4D
radar used in the IRIS-T SLM system.

.. todo::

   Document the offline/local map tile server setup used as a fallback by
   the scenario editor and frontend when public tile servers (OpenStreetMap,
   ArcGIS) are unreachable.
