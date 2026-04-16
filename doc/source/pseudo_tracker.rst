.. _pseudo_tracker:

Pseudo tracker
==============

There is either no or exactly one track per target.

Track initialisation
--------------------

A new track is initialised in the following cases:

- There is at least one monostatic radar detection for the target.

    If multiple mononostatic detections are present, only the one with the
    shortest monostatic range is considered. That detection (inlcuding uncertainty)
    is transformed to ECEF coordinates and initialises the track.
- There are at least three PCL detections for the target.

    A simplified detection model in ECEF coordinates is built. It is a Gaussian
    distribution, centered at the true target position and with
    standard deviations derived from the detections' bistatic range uncertainty.
    A point in ECEF space is sampled from this detection model and initialises
    the track.

Track update
------------

An existing track is updated in the following cases, listed in decreasing preference:

- There is at least one monostatic radar detection for the target.

    If multiple mononostatic detections are present, only the one with the
    shortest monostatic range is considered.
    
    That detection (inlcuding uncertainty) is transformed to ECEF coordinates
    and updates the track.

- There is at least one PCL detection for the target.

    If there are multiple PCL detections, we select the one with the shortest
    bistatic range.

    We assume that the target speed has not changed since the previous track update,
    i. e. that the true next position :math:`\mathbf{x}` lies on a sphere of
    radius :math:`\rho = v \Delta t`, where the quantity :math:`v` represents
    the target speed and the quantity :math:`\Delta t` represents the time
    since the latest track update.

    The ellipsoid surface is sampled and the distance of each point to the sphere's
    surface is calculated. The closest point is chosen as the plot to be added to the track. [#efficiency]_

    .. figure:: img/pseudo_tracker_pcl.svg

Track deletion
--------------

A track is deleted if more than `patience` iterations have passed since the
latest update.


.. [#efficiency] There might be more efficient approaches, e. g. by deriving closed
   form solutions. We keep it simple to implement because it might be enough and
   the pseudo tracker is an approximation in the first place.


Example
=======

Consider a simple example situation with three transmitters (blue) and one receiver (black).

.. raw:: html

    <iframe
        src="/home/user/Documents/theia_backend/doc/build/html/_static/pcl_example.html"
        width="100%"
        height="400px"
        style"border:none;display=block;"
    >
    </iframe>

The minimum detectable radar cross section (RCS) can be calculated for each
position within the rectangle and each (transmitter, receiver) pair. A hypothetical
target of RCS = :math:`1 m^2` can be detections in regions where the minimum detectable RCS
is :math:`\leq 1 m^2`, which is plotted in the second row.

.. image:: _static/pcl_example_coverages.png

A track can be initialised if at least three PCL sensors are able to detect the target
(almost) at the same time. That is, the region of PCL track init is given by the
logical AND of each detection map.

.. image:: _static/pcl_example_init_coverage.png

This is the example used in the unit tests. The pseudo track will not initialise
a track for a target travelling within the red zone.
However, it will initialise a track if the target is within the green region.
