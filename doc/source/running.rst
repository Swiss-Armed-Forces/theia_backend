Running the server
====================

Theia exposes its simulation over a FastAPI server run with `uvicorn`. There
are two ways to start it, depending on what you want to do.

Standalone mode
-----------------

.. code-block:: bash

   poetry run python scripts/run_server.py

Starts a server backed by an empty "dummy" simulation: no scenario is
loaded and there are no controllers. This is the mode used by the
**scenario editor**, which only needs the server for terrain models, FM
transmitter data, default sensor configurations, and coverage calculations
— not for running an actual simulation.

Running a scenario interactively
-----------------------------------

.. code-block:: bash

   poetry run python scripts/simulate_scenario.py <scenario_file> <output_file> --interactive

Loads a scenario file and exposes it live over the same API as it runs.
This is the mode used by the **live visualisation frontend**, which polls
endpoints such as ``/situational_picture`` and ``/ground_truth`` as the
simulation advances.

Without ``--interactive``, the scenario is instead run headless to
completion, writing its output to ``<output_file>`` without starting a
server at all — useful for batch runs and automated analysis.

Startup time
-------------

Terrain data is loaded before the server starts listening, so expect a
delay (loading preprocessed hierarchical bounding boxes for the scenario's region)
before uvicorn prints its usual startup log. Wait for that message, or poll ``/docs``,
before connecting a client.

Stopping the server
---------------------

Stop the process with ``Ctrl+C`` (or by killing it). There is no separate
shutdown endpoint.

Ports
------

Both modes bind to ``127.0.0.1:8000`` by default. This matches what the
scenario editor and live frontend expect out of the box.
