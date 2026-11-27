#!/bin/bash
# Maximum speedup of the game demo on the interactive server, with or without
# the GUI.
#
# Usage
# -----
#     scripts/benchmark_game_demo.sh nogui|gui [DURATION_S] [OUT_DIR]
#     SPY=1 scripts/benchmark_game_demo.sh gui 60     # also profile with py-spy
#
# The server runs scenarios/player_actions_test at speedup 1e6, i.e. as fast
# as it can. Every 5 s, the game time is read from /time; the speedup is the
# game time per wall-clock time. Prints "wall_s game_s" lines.
#
# With "gui", a headless Chromium (Playwright) keeps the frontend open, which
# polls the server like a real user's browser. The frontend's dev server must
# be running (cd ../theia_frontend && npm run dev).
#
# With SPY=1, the server runs as a child of py-spy (no ptrace permission
# needed); the profile is written to OUT_DIR/spy_MODE.txt (collapsed stacks).
# py-spy pauses the process for every sample: do not use the speedup of such a
# run.
#
# Environment: PYTHON (default: the Poetry environment's python),
# FRONTEND_DIR (default: ../theia_frontend), FRONTEND_URL (default:
# http://localhost:5173). Port 8000 must be free.

set -euo pipefail

MODE=${1:?"Usage: $0 nogui|gui [DURATION_S] [OUT_DIR]"}
DUR=${2:-120}
OUT=${3:-$(mktemp -d)}
case $MODE in nogui | gui) ;; *) echo "Unknown mode $MODE" >&2; exit 1 ;; esac

BACKEND_DIR=$(cd "$(dirname "$0")/.." && pwd)
FRONTEND_DIR=${FRONTEND_DIR:-$BACKEND_DIR/../theia_frontend}
FRONTEND_URL=${FRONTEND_URL:-http://localhost:5173}
PYTHON=${PYTHON:-$(cd "$BACKEND_DIR" && poetry env info -p)/bin/python}
SCENARIO=scenarios/player_actions_test/player_actions_test.json
mkdir -p "$OUT"

if lsof -ti:8000 -sTCP:LISTEN >/dev/null; then
    echo "Port 8000 is in use" >&2
    exit 1
fi

RUN=("$PYTHON")
if [ -n "${SPY:-}" ]; then
    RUN=("$(dirname "$PYTHON")/py-spy" record -r 100 --format raw -o "$OUT/spy_$MODE.txt" -- "$PYTHON")
fi
(cd "$BACKEND_DIR" && exec "${RUN[@]}" scripts/simulate_scenario.py "$SCENARIO" \
    "$OUT/log_$MODE.json" --interactive >"$OUT/server_$MODE.log" 2>&1) &
SERVER=$!

cleanup() {
    local pid
    pid=$(pgrep -f "^$PYTHON scripts/simulate_scenario.py" | head -1 || true)
    if [ -n "$pid" ]; then
        kill -INT "$pid" 2>/dev/null || true
        sleep 5
        kill -9 "$pid" 2>/dev/null || true
    fi
    wait "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

timeout 120 bash -c 'until curl -sf localhost:8000/time >/dev/null; do sleep 1; done'
curl -sf -X POST "localhost:8000/speedup?speedup_factor=1000000" >/dev/null

GUI=
if [ "$MODE" = gui ]; then
    # Run from the frontend directory: Node resolves "playwright" from there.
    (cd "$FRONTEND_DIR" && node --input-type=module -e '
        import { chromium } from "playwright";
        const [url, seconds] = process.argv.slice(1);
        const browser = await chromium.launch();
        const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
        let errors = 0;
        page.on("console", (m) => { if (m.type() === "error") errors++; });
        await page.goto(url);
        await page.waitForSelector(".leaflet-container");
        await page.waitForTimeout(Number(seconds) * 1000);
        console.log(`GUI console errors: ${errors}`);
        await browser.close();
    ' "$FRONTEND_URL" $((DUR + 10)) >"$OUT/gui.log" 2>&1) &
    GUI=$!
    sleep 8 # Let the GUI load before measuring.
fi

# No /resume: the director starts the simulation by itself, and a second
# resume would start a second simulation thread.
read_time() { curl -sf localhost:8000/time | tr -d '"'; }
T0=$(read_time)
W0=$(date +%s.%N)
echo "wall_s game_s"
for _ in $(seq 1 $((DUR / 5))); do
    sleep 5
    T=$(read_time)
    W=$(date +%s.%N)
    python3 -c "
import datetime as d
t0 = d.datetime.fromisoformat('$T0'.replace('Z', '+00:00'))
t = d.datetime.fromisoformat('$T'.replace('Z', '+00:00'))
print(f'{$W - $W0:.1f} {(t - t0).total_seconds():.0f}')"
done
curl -sf -X POST localhost:8000/pause >/dev/null

if [ -n "$GUI" ]; then
    wait "$GUI" || true
    cat "$OUT/gui.log"
    echo "/situational_picture requests: $(grep -c 'POST /situational_picture' "$OUT/server_$MODE.log" || true)"
fi
echo "Output: $OUT"
