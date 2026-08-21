#!/usr/bin/env bash
# Run the frozen coordinate-transformation correctness tests, then the
# performance benchmark, in one shot. Exits non-zero if either test file
# fails, so it's safe to chain in CI or scripting.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-/home/claude_code/.cache/pypoetry/virtualenvs/theia-iLCo1Bwa-py3.12/bin/python}"

echo "=== ruff check src/theia/coordinates.py ==="
"$PYTHON" -m ruff check src/theia/coordinates.py

echo
echo "=== tests/test_coordinate_transformations_pyproj_reference.py ==="
"$PYTHON" -m unittest tests.test_coordinate_transformations_pyproj_reference -v

echo
echo "=== tests/test_ecef_to_enu_pyproj_reference.py ==="
"$PYTHON" -m unittest tests.test_ecef_to_enu_pyproj_reference -v

echo
echo "=== tests/test_azimuth_elevation_pyproj_reference.py ==="
"$PYTHON" -m unittest tests.test_azimuth_elevation_pyproj_reference -v

echo
echo "=== tests/test_coordinate_tansformations.py ==="
"$PYTHON" -m unittest tests.test_coordinate_tansformations -v

echo
echo "=== tests/test_elevation_angle.py ==="
"$PYTHON" -m unittest tests.test_elevation_angle -v

echo
echo "=== scripts/benchmark_coordinate_transformations.py ==="
"$PYTHON" scripts/benchmark_coordinate_transformations.py

echo
echo "=== scripts/benchmark_ecef_to_enu.py ==="
"$PYTHON" scripts/benchmark_ecef_to_enu.py

echo
echo "=== scripts/benchmark_azimuth_elevation.py ==="
"$PYTHON" scripts/benchmark_azimuth_elevation.py
