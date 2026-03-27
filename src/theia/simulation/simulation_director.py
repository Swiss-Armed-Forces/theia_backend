import threading

from theia.simulation.simulator import Simulator


class SimulationDirector:
    def __init__(self, simulator: Simulator):
        self._simulator = simulator
        self._is_paused: bool = False
        self.resume()

    def resume(self):
        self._is_paused = False

        def runner():
            while not self._is_paused:
                self._simulator.advance()

        sim_thread = threading.Thread(
            target=runner,
            daemon=False,
        )
        sim_thread.start()

    def pause(self):
        self._is_paused = True

    def is_paused(self) -> bool:
        return self._is_paused