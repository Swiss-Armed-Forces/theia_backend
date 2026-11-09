import threading

from theia.simulation.simulator import Simulator


class SimulationDirector:
    def __init__(self, simulator: Simulator):
        self._simulator = simulator
        self._is_paused: bool = False
        self._is_stopped: bool = False
        self._sim_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        """Serialises resume and stop."""
        self.resume()

    def resume(self):
        with self._lock:
            if self._is_stopped:
                return
            self._is_paused = False

            def runner():
                while not self._is_paused and self._simulator.advance():
                    pass

            self._sim_thread = threading.Thread(
                target=runner,
                daemon=False,
            )
            self._sim_thread.start()

    def pause(self):
        self._is_paused = True

    def is_paused(self) -> bool:
        return self._is_paused

    def stop(self):
        """
        End the simulation properly: the listeners are notified (e. g. the
        log file is written) before this method returns.

        Works while running and while paused. A stopped simulation cannot be
        resumed.
        """
        with self._lock:
            self._is_stopped = True
            self._simulator.request_stop()
            if self._sim_thread is not None:
                # Ends after at most one more iteration.
                self._sim_thread.join()
            if not self._simulator.has_ended():
                # Paused: nobody else advances the simulation any more.
                self._simulator.advance()
            self._is_paused = True

    def is_stopped(self) -> bool:
        return self._is_stopped
