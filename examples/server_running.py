import numpy as np
import uvicorn
from theia.simulation.factories.bodensee_monostatic import BodenseeMonostaticFactory
from theia.simulation.factories.performance_demo import PerformanceDemoFactory
from theia.simulation.factories.uetliberg_opensky import (
    UetlibergOpenskySimulatorFactory,
)
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector
from theia.terrain import SrtmTerrainModel

print("Initialise...")

################################################
# Setup the simulation.
################################################
rng = np.random.Generator(np.random.PCG64(seed=4054080))
terrain = SrtmTerrainModel()
factory = PerformanceDemoFactory(rng, terrain)
# factory = UetlibergOpenskySimulatorFactory()
# factory = BodenseeMonostaticFactory(rng)
simulator, buffer = factory.build_simulator(True, rng, terrain)

################################################
# Setup the server.
################################################
app = create_app(buffer, SimulationDirector(simulator))

print("Start simulation...")

uvicorn.run(app, host="127.0.0.1", port=8000)
