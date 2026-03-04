import cProfile

from theia.simulation.logging import LogLoader
from theia.simulation.tracking import MonostaticSingleSensorTracker

# Load data.
loader = LogLoader("log_opensky.json")

# Select data subset for testing.
RADAR_ID = 0
TARGET_ID = 66
TARGET_ID2 = 3

detections = [
    d
    for d in loader.blue_monostatic_radar_detections
    if d.metadata["radar_id"] == RADAR_ID
    and d.metadata["target_id"] in [TARGET_ID, TARGET_ID2, -2]
]

radar = next(
    radar for radar in loader.blue_monostatic_radars if radar.receiver.id == RADAR_ID
)
ground_truths = [
    loader.red_target_ground_truth[TARGET_ID],
    loader.red_target_ground_truth[TARGET_ID2],
]

times = []
for gt in ground_truths:
    times.extend([state.timestamp for state in gt])
times = sorted(list(set(times)))

# Track.
times = [state.timestamp for state in ground_truths[0]]
tracker = MonostaticSingleSensorTracker()

profiler = cProfile.Profile()
profiler.enable()

for time in times[:200]:
    detections_at_time = set([d for d in detections if d.timestamp == time])
    if (len(detections_at_time)) == 0:
        continue
    tracker.add_detections(detections_at_time)

profiler.disable()
profiler.dump_stats("profile.prof")
