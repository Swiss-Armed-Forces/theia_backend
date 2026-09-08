Architecture
=============

System overview
-----------------

Theia is split across three repositories that communicate exclusively
through this backend's HTTP API (see :doc:`../running` and :doc:`../api`):

- **theia_backend** (this repository) — the simulation engine and API
  server. It can run standalone (serving terrain models and default
  sensor configurations, with no scenario loaded) or with a scenario
  loaded, in which case it simulates and exposes live results.
- **theia_scenario_editor** — a GUI for building a scenario (sensor and
  effector placement, order of battle). It talks to the backend running
  in standalone mode to look up terrain models, transmitters of opportunity
  for PCL, and default configurations, and produces a JSON-based order
  of battle (orbat) file consumed by the backend.
  Currently, the scenario editor builds an order of battle for one party.
  The full JSON-based scenario file is created by merging a RED and a BLUE
  orbat file using a script from the backend repo.
- **theia_frontend** — a GUI for live visualisation of a running
  simulation. It polls the backend's API (situational picture, ground
  truth, GeoJSON layers) while a scenario is being simulated
  interactively.

Typical workflow
----------------

1. Build an orbat for the BLUE forces in the scenario editor.
2. Build an orbat for the RED forces in the scenario editor.
3. Merge the orbat files into a single scenario file through the backend
   (``scripts/build_scenario_file.py``)
4. Simulate the scenario file through the backend
   (``scripts/simulate_scenario.py --interactive``)
5. Observe it live in the frontend.

Backend architecture
----------------------

A theia simulation is an instance of the class ``Simulator``. This class embodies
the control flow and glue between the building blocks. Its philosophy follows
the delegator pattern.

The main task of the ``Simulator`` is to advance target state ("ground truth")
and calculate raw detections including clutter. These are passed to a subclass of
``AbstractTracker``, which generates tracks. The enemy tracks together with
known friendly positions form the situational picture, i. e. an object of type
``theia.types.SituationalPicture``. The situational picture, in turn,
informs the decisions of ``Controller`` s, which implement the behaviour logic.

Situational awareness: Trackers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is one tracker for blue and one for red. The composite pattern can be
used to unify multiple underlying trackers, e. g. one tracker per monostatic radar.
This architecture allows for the creation of complex sensor networks and fine-grained
control over how information is fused.

Behaviour: Controllers
^^^^^^^^^^^^^^^^^^^^^^^^

There is one controller for blue and one for red. That global controller is
responsible for the entire behaviour of the simulated entities.
The composite pattern enables the creation of complex command chains by nesting
controllers. For example, the bottom level could represent a single weapon system
(e. g. GBAD launcher) or drone. That "GBAD launcher controller" has a set of basic
building blocks such as the following:

- "engage the assigned target" (implemented)
- "only activate monostatic radar once N passive radar detections have been collected"
- "defend critical infrastructure X"

A drone controller might have building blocks such as:

- "follow a trajectory through waypoints in 3D space" (implemented)
- "follow a 2D trajectory using terrain following"
- "loiter until a target comes into your range, then engage"

A higher level command controller could then be used to coordinate and issue orders to the lower level
controllers. This approach can be nested to construct complex orbats.

Finally, a controller might represent human input.

Note that most of the behaviour mentioned above has not been implemented yet. These examples showcase
a possible future version of Theia, though much work is needed to get there.

Readout & Visualisation: Loggers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Simulator`` contains a single instance of ``AbstractSimulationLogger``.
There are two main use cases for the loggers.

- Persist results
  The ground truth, events and detections are logged to a file for later analysis.
- Expose simulation state for live visualisation
  The current state of the simulation is exposed to a webserver, which streams
  it to a visualisation software. This is how the live frontend's data is produced.

The composite pattern allows for the combination multiple loggers and the filter pattern
can be used to configure different logging frequencies for different types of information.
In the future, the logging interface could be used as part of an interoperability layer.

Currently, the following entities are logged:

- ``SituationalPicture`` for blue and red (by default not enabled to keep file size small)
- ``Snapshot`` (represents the ground truth)
- ``MonostaticRadarDetection`` (for developing new trackers)
