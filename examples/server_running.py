import numpy as np
import uvicorn
from theia.simulation.factories.bodensee_monostatic import BodenseeMonostaticFactory
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector

print("Initialise...")

################################################
# Setup the simulation.
################################################
rng = np.random.Generator(np.random.PCG64(seed=4054080))
factory = BodenseeMonostaticFactory(rng)
simulator, buffer = factory.build_simulator(True, rng)

################################################
# Setup the server.
################################################
app = create_app(buffer, SimulationDirector(simulator))

print("Start simulation...")

uvicorn.run(app, host="127.0.0.1", port=8000)
