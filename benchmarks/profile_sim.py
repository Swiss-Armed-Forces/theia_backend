import cProfile

import numpy as np
from tqdm import tqdm

from theia.simulation.factories.performance_demo import PerformanceDemoFactory
from theia.simulation.logging import FileLogger


# simulator, buffer = load_single_target_from_Bodensee_simulator(interactive=False)
rng = np.random.Generator(np.random.PCG64(seed=4054080))
factory = PerformanceDemoFactory(rng)
simulator, buffer = factory.build_simulator(False, rng)
listener = FileLogger("log.json")
simulator.set_listener(listener)

profiler = cProfile.Profile()
profiler.enable()

for _ in tqdm(range(650)):
    simulator.advance()

profiler.disable()
profiler.dump_stats("profile__simulator_advance.prof")

listener.on_end()
