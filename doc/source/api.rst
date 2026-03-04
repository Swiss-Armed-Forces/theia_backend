API Architecture
================

Introduction
------------
A theia simulation is an instance of the class ``Simulator``. This class embodies
the control flow and glue between the building blocks. Its philosophy follows
the delegator pattern.

The main task of the ``Simulator`` is to advance target state ("ground truth")
and calculate raw detections including clutter. These are passed to a subclass of
``AbstractTracker``, which generates tracks. The enemy tracks together with
known friendly positions form the situational picture, i. e. an object of type
``theia.types.SituationalPicture``. The situational picture, in turn,
informs the decisions of ``Controller`` s.

Situational awareness: Trackers
-------------------------------

There is one tracker for blue and one for red. The composite pattern can be
used to unify multiple underlying trackers, e. g. one tracker per monostatic radar.

Behaviour: Controllers
----------------------

There is one controller for blue and one for red. That global controller is
responsible for the entire behaviour of the simulated entities.
The composite pattern enables the creation of complex command chains by nesting
controllers. For example, the bottom level could represent a single weapon system
(e. g. main battle tank) or soldier. That "soldier controller" has a set of basic
building blocks such as the following:

- "move on streets to position a"
- "hold zone b"
- "conquer zone c"
- "follow entity d and provide fire cover for it"
- "bomb target entity e"
- "fly using terrain following to position f"
- "fly at fix altitude g to position h"

A higher level command controller could then be used to coordinate and issue orders to the soldier
controller. This approach can be nested to construct complex orbats.

Finally, a controller might represent human input.

Readout & Visualisation: Loggers
--------------------------------

The ``Simulator`` contains a single instance of ``AbstractSimulationLogger``.
In the simplest case, the logger does nothing. Another implementation might
print the current state to the command line or a file. There might also be the
option to expose the current state of the simulation to a webserver, which streams
it to a sophisticated visualisation in Javascript or similar.

The composite pattern allows for the combination multiple loggers and the filter pattern
can be used to configure different logging frequencies for different types of information.

Currently, the following entities are logged:

- ``SituationalPicture`` for blue and red
- ``Snapshot`` (represents the ground truth)
- ``MonostaticRadarDetection`` (for developing new trackers)
