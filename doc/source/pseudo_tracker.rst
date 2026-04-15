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