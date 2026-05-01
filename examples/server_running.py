import uvicorn

from theia.simulation.predefined_simulations import (
    load_uetliberg_opensky_simulator,
    load_uetliberg_single_target_simulator_pcl,
)
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector

print("Initialise...")

################################################
# Setup the simulation.
################################################
simulator, buffer = load_uetliberg_opensky_simulator()
# simulator, buffer = load_uetliberg_single_target_simulator_pcl(
#     bistatic_range_uncertainty=100.0,
# )

################################################
# Setup the server.
################################################
app = create_app(buffer, SimulationDirector(simulator))

print("Start simulation...")

uvicorn.run(app, host="127.0.0.1", port=8000)
