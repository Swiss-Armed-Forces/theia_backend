.. _pseudo_tracker:

Pseudo tracker
==============

.. warning::

   The pseudo tracker does NOT aim to be usable in a real-world setup.
   It's purpose is to emulate a real tracker, including uncertainty, for simulation
   purposes. It explicitly uses ground truth information to speed up and
   simplify the tracking process.

.. _schematic:

.. figure:: img/pseudotracker.svg
   :alt: Schematic depiction of the pseudo tracker algorithm.

   Schematic depiction of the pseudo tracker algorithm.

Assumptions
-----------

- There is either no or exactly one track per target.
- Data association (i. e. assigning detections to target IDs and removing clutter)
  is perfect.
- Tracking happens in Cartesian ECEF space.

Description
-----------

A schematic of the algorithm is depicted in Fig. :numref:`schematic`.

A new track is initialized only if the target position can be known exactly
(modulo uncertainty). This is described in the first row. Whenever there are
monostatic detections available, direction and range information is known (case 1).
When there are multiple monostatic detections, the one with the shortest range is
used because shorter range implies smaller error due to angular uncertainty.

When there are at least three PCL detections, the target lies at the intersection
of the ellipsoids (case 2). Instead of computing it exactly, we propagate
bistatic range uncertainty to ECEF space and sample the detected position in
ECEF space. This allows to combine the detections with monostatic and PET detections.
The ECEF target position distribution is modelled as a Gaussian around the
ground truth position. The standard deviation is the maximum standard deviation
of all bistatic range uncertainties.

A similar situation is used when there are at least two PET detections (case 3).
The only difference is that the uncertainty in ECEF space is estimated using
the maximum uncertainty of the direction cone at ground truth range.

The case of 2 PCL detections and 1 PET detection is handled in an analogue way.
The uncertainties of the PCL-only and PET-only cases are combined by taking
the maximum value.

The final two cases do not allow to initialize a track since they do not detect
a target unambiguously. However, they can still be used to update an existing
track assuming perfect association.

.. Consider two PCL detections with measured bistatic ranges
.. :math:`\rho_i \sim \mathcal{N}(R_i, \sigma_i), \quad i = 1, 2` distributed normally
.. around the true values :math:`R_i`.
.. Since both measurement processes are independent, the probability of obtaining
.. a sample for the given true values :math:`\mathbf{R}` is given by

.. .. math::
    
..     p(\mathbf{\rho} | \mathbf{R}) = \frac{1}{4 \pi \sqrt{\sigma_1 \sigma_2}} \exp\left(-\frac{(\mathbf{\rho_1} - \mathbf{R_1})^2}{2 \sigma_1^2}\right) \exp\left(-\frac{(\mathbf{\rho_2} - \mathbf{R_2})^2}{2 \sigma_2^2}\right).

.. Using Bayes' law:

.. .. math::

..     p(\mathbf{R} | \mathbf{\rho}) = \frac{p(\mathbf{\rho} | \mathbf{R}) p(\mathbf{R})}{p(\mathbf{\rho})}

Given two PCL detections and no other detections (case 5), it is possible to
sample one ellipse and consider the points within a threshold distance to the
surface of the other ellipsoid. The threshold is given by the sum of the two ellipses'
bistatic range uncertainty.

.. Track initialisation
.. --------------------

.. A new track is initialised in the following cases:

.. - There is at least one monostatic radar detection for the target.

..     If multiple mononostatic detections are present, only the one with the
..     shortest monostatic range is considered. That detection (inlcuding uncertainty)
..     is transformed to ECEF coordinates and initialises the track.
.. - There are at least three PCL detections for the target.

..     A simplified detection model in ECEF coordinates is built. It is a Gaussian
..     distribution, centered at the true target position and with
..     standard deviations derived from the detections' bistatic range uncertainty.
..     A point in ECEF space is sampled from this detection model and initialises
..     the track.

.. Track update
.. ------------

.. An existing track is updated in the following cases, listed in decreasing preference:

.. - There is at least one monostatic radar detection for the target.

..     If multiple mononostatic detections are present, only the one with the
..     shortest monostatic range is considered.
    
..     That detection (inlcuding uncertainty) is transformed to ECEF coordinates
..     and updates the track.

.. - There is at least one PCL detection for the target.

..     If there are multiple PCL detections, we select the one with the shortest
..     bistatic range.

..     We assume that the target speed has not changed since the previous track update,
..     i. e. that the true next position :math:`\mathbf{x}` lies on a sphere of
..     radius :math:`\rho = v \Delta t`, where the quantity :math:`v` represents
..     the target speed and the quantity :math:`\Delta t` represents the time
..     since the latest track update.

..     The ellipsoid surface is sampled and the distance of each point to the sphere's
..     surface is calculated. The closest point is chosen as the plot to be added to the track. [#efficiency]_

..     .. figure:: img/pseudo_tracker_pcl.svg

Track deletion
--------------

A track is deleted if more than `patience` iterations have passed since the
latest update.


.. .. [#efficiency] There might be more efficient approaches, e. g. by deriving closed
..    form solutions. We keep it simple to implement because it might be enough and
..    the pseudo tracker is an approximation in the first place.


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
