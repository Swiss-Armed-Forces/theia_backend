import uvicorn

from theia.simulation.predefined_simulations import (
    build_single_radar_single_target,
    load_single_target_from_Bodensee_simulator,
    load_uetliberg_opensky_simulator,
    load_uetliberg_single_target_simulator_pcl,
)
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector

print("Initialise...")

################################################
# Setup the simulation.
################################################
# simulator, buffer = load_uetliberg_opensky_simulator()
# simulator, buffer = load_uetliberg_single_target_simulator_pcl(
#     bistatic_range_uncertainty=100.0,
# )
# simulator, buffer = load_single_target_from_Bodensee_simulator()
simulator, buffer = build_single_radar_single_target(interactive=True)

################################################
# Setup the server.
################################################
app = create_app(buffer, SimulationDirector(simulator))

print("Start simulation...")

uvicorn.run(app, host="127.0.0.1", port=8000)
