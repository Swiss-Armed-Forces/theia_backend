"""
Export the REST API's OpenAPI schema, e. g. for generating frontend types:

    python scripts/export_openapi.py ../theia_frontend/src/hooks/openapi.json
"""

import argparse
import json
import pathlib

from theia.simulation.server import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Theia's OpenAPI schema")
    parser.add_argument("output_file", type=pathlib.Path)
    args = parser.parse_args()

    # Generating the schema does not touch the buffer or the director.
    app = create_app(buffer=None, director=None)
    with open(args.output_file, "w") as file:
        json.dump(app.openapi(), file, separators=(",", ":"))
