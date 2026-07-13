import cProfile

import numpy as np
from tqdm import tqdm

from theia.simulation.factories.performance_demo import PerformanceDemoFactory
from theia.simulation.theia_logging import FileLogger
from theia.terrain import SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel, HbvTree


# simulator, buffer = load_single_target_from_Bodensee_simulator(interactive=False)
rng = np.random.Generator(np.random.PCG64(seed=4054080))
terrain = SrtmTerrainModel()
terrain = FastSrtmModel(
    tree=HbvTree.load("tree_lat47:48_lon7:8.zip"),
    srtm_model=terrain,
)
factory = PerformanceDemoFactory(rng, terrain)
simulator, buffer = factory.build_simulator(False, rng, terrain)
listener = FileLogger("log.json")
simulator.set_listener(listener)

profiler = cProfile.Profile()
profiler.enable()

for _ in tqdm(range(650)):
    simulator.advance()

profiler.disable()
profiler.dump_stats("profile__simulator_advance.prof")

listener.on_end()
