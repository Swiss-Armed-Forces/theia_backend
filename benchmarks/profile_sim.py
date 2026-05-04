import cProfile

from tqdm import tqdm

from theia.simulation.logging import FileLogger
from theia.simulation.predefined_simulations import (
    load_single_target_from_Bodensee_simulator,
)


simulator, buffer = load_single_target_from_Bodensee_simulator(interactive=False)
listener = FileLogger("log.json")
simulator.set_listener(listener)

profiler = cProfile.Profile()
profiler.enable()

for _ in tqdm(range(650)):
    simulator.advance()

profiler.disable()
profiler.dump_stats("profile__simulator_advance.prof")

listener.on_end()
