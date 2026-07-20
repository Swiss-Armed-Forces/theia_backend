import numpy as np
from tqdm import tqdm
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.factories.zurich_airport import ZurichAirportScenarioFactory
from theia.simulation.theia_logging import FileLogger
from theia.terrain import SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel, HbvTree

print("Initialise...")

################################################
# Setup the simulation.
################################################
rng = np.random.Generator(np.random.PCG64(seed=4054080))
tree = HbvTree.load("tree_lat46:49_lon7:11_subsamplestride2.zip")
terrain = FastSrtmModel(tree=tree, srtm_model=SrtmTerrainModel(), t_min=30)
terrain = SrtmTerrainModel()
damage_model = UniformDamageModel(1.0, rng)  # Every shot kills.
factory = ZurichAirportScenarioFactory(rng, terrain)
simulator, buffer = factory.build_simulator(False, rng, terrain, damage_model)
logger = FileLogger("log.json")
logger.register_simulator(simulator)
simulator.set_listener(logger)

################################################
# Run the simulation.
################################################
for _ in tqdm(range(1000)):
    simulator.advance()
logger.on_end()
