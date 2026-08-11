<img src="doc/source/_static/logo.png" alt="Logo of Theia" width="256" />

Theia is an agent-based simulation code for integrated air defense (IAD) by the Swiss Armed Forces. It is inspired by and loosely based on the open source code [openBURST](https://github.com/Swiss-Armed-Forces/openburst).

# What is Theia?

Theia is an open source framework for simulating integrated air defence to assess the consequences of an operational concept for sensors and effectors, compare multiple variants and assess their performances against various hostile forces.

Existing open source simulation frameworks for defense focus on one of two use cases.
The first type focus on simulating physics and sensors in very specific situations, while considering
decision making setting the stage, but not being modelled. The second type, on the other hand,
simplifies the sensor network, usually omitting tracking altogether, for the sake
of simulating complex behaviour and decision making.

Theia aims to fill this gap and is, to the best of our knowledge, the only
framework that fulfills all of the following properties:

- It is open source.
- It detections for complex heterogeneous sensor networks of multiple types, including active radar, PCL, PET and visual sensors.
- It fuses information from various sensors and prior knowledge into a unified situational picture using tracking algorithms provided by frameworks such as [Stone Soup](https://github.com/dstl/Stone-Soup/).
- It is capable of modelling complex high-level behaviour with incomplete information, informed by the simulated situational picture ("if you cannot see it, you cannot react to it"). Ex.: Target assignment, emission control, movement, ...
- It provides a GUI, but can be run headless.

# What questions does Theia answer?

All of these questions can be answered using the same simulation output.

- **How effective is the sensor placement?**
  - Theia calculates coverage maps.
  - Theia simulates hostile forces and simultaneously friendly sensor detections.
  - Theia fuses the information into tracks.
  - Theia simulates effectors that fight the identified targets.
  - Output:
    - Geometric coverage maps
    - Detection performance ("x% of enemy ballistic missiles were tracked with error <y%")
    - Resilience ("what if x% of sensors are out of order?")
- **How effectice is the effector placement?**
  - Theia calculates effector range maps.
  - Theia simulates detection & information fusion.
  - Theia simulates effectors probabilistically.
  - Output:
    - Geometric coverage maps
    - Battle performance ("x% of detected drones are eliminated before approaching closer than y meters")
    - Ammo stockpile requirements
    - Ammo throughput requirements (replenishment requirements)
- **How do decision for the sensor placement propagate to battle outcome?**
- **Future Force Design**
  - Pros and cons of various force package configurations
  - What is the tactical and operational benefit of a dollar spent on system x in scenario y in terms of KPIs?
  - Where are the bottle necks? Is it sensor placement, effector placement, ammo stockpiling, replenishment, ...?
  - How well do various force package configurations perform in different scenarios?
  - Tradeoff analysis: Which goals go hand in hand, which goals ae mutually exclusive?<br />
    Example: Is it possible to fend off ballistic missiles while optimising for drone defense?

# Physics model

- Terrain is considered. By default, [SRTM terrain data](https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1.003) are used, so earth curvature is taken into account.
- Radar radiation is assumed to travel in straight lines; refraction is neglected (k = 1 for the earth radius).
- Propagation effects are neglected.
- Detections are simulated using the radar equation and SNR thresholding, combined with Doppler thresholding.


# Installation

First, [install poetry](https://python-poetry.org/docs/#installation).

Then, the repository can be installed as follows:

```bash
poetry install
```

# Generating doc

Run the following commands from the repo root directory:

```bash
cd doc/
make html
```

You will find the HTML documentation at `doc/build/html/index.html`.

# Running unit tests

```bash
python -m unittest discover tests
```

# Publishing a new release

The package version is inferred from the git tags. We use semantic versioning.

1. ``git tag v1.2.3`` (insert your version!)
2. ``git push; git push --tags``
3. ``poetry install``

You can test whether the correct version is picked up using ``poetry version``.
