# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

import folium
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from theia.detection.pcl import PclDetector
from theia.mapping import RadarMap
from theia.terrain import elevationAt
from theia.test_data import load_pcl_example_sensors


project = "theia"
copyright = "2026, Jürg Huber"
author = "Jürg Huber"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
]
bibtex_bibfiles = ["refs.bib"]

templates_path = ["_templates"]
exclude_patterns = []

html_logo = "_static/logo.png"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def save_pcl_example_setup(app):
    example_sensors, lats, lons, alts = load_pcl_example_sensors()
    map = RadarMap(
        sensors={str(sensor.id): sensor for sensor in example_sensors}
    ).to_map()
    folium.Rectangle(
        bounds=[[lats.min(), lons.min()], [lats.max(), lons.max()]]
    ).add_to(map)
    folium.Marker(
        (example_sensors[0].receiver.lat, example_sensors[0].receiver.lon),
        tooltip="Rx",
        icon=folium.Icon(color="black"),
    ).add_to(map)
    map.location = (47.3356, 8.4707)
    static_dir = os.path.join(app.outdir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    map.save(f"{static_dir}/pcl_example.html")


def save_pcl_coverage_plots(app):
    static_dir = os.path.join(app.confdir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    path = f"{static_dir}/pcl_example_coverages.png"
    path2 = f"{static_dir}/pcl_example_init_coverage.png"

    # Do not regenerate the plots every time because it takes a while.
    if os.path.exists(path):
        return
    
    example_sensors, lats, lons, alts = load_pcl_example_sensors()

    detector = PclDetector()

    minimum_detectable_rcs = np.empty(
        (len(lats), len(lons), len(alts), len(example_sensors)), dtype=np.float32
    )
    minimum_detectable_rcs[:, :, :] = np.nan
    for s, sensor in enumerate(example_sensors):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                for k, alt in enumerate(alts):
                    if alt <= elevationAt(lat, lon):
                        continue
                    minimum_detectable_rcs[i, j, k, s] = (
                        detector.minimum_detectable_rcs_vector(
                            sensor.receiver,
                            sensor.transmitter,
                            np.array([[lat, lon, alt]]),
                        )[0]
                    )

    RCS = 1.0

    fig, axes = plt.subplots(
        figsize=(4 * 8, 2 * 4.5), ncols=4, nrows=2, width_ratios=[1.0, 1.0, 1.0, 0.05]
    )

    label_skip = 20

    for i, sensor in enumerate(example_sensors):
        ax = axes[0, i]
        img = ax.imshow(minimum_detectable_rcs[:, :, 0, i], vmin=0, vmax=3)
        ax.set_title(f"Minimum detectable RCS for sensor {sensor.id}")

        ax = axes[1, i]
        cmap = mcolors.ListedColormap(["#eb442c", "#0c6b37"])
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(minimum_detectable_rcs[:, :, 0, i] <= RCS, cmap=cmap, norm=norm)
        ax.set_title(f"Detection zones for RCS = {RCS} $m^2$ (sensor {sensor.id})")

    for ax in axes[:, :3].flatten():
        ax.set_xlabel("lon [°]")
        ax.set_ylabel("lat [°]")
        ax.set_xticks(np.arange(len(lons))[::label_skip])
        ax.set_yticks(np.arange(len(lats))[::label_skip])
        ax.set_xticklabels([f"{lon:.3f}" for lon in lons][::label_skip])
        ax.set_yticklabels([f"{lat:.3f}" for lat in lats][::label_skip])
        ax.tick_params(axis="x", rotation=90)
        ax.invert_yaxis()

    axes[1, 3].remove()
    fig.colorbar(img, cax=axes[0, 3])
    fig.tight_layout()
    fig.savefig(path, transparent=True)


    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.imshow(
        np.all(minimum_detectable_rcs[:, :, 0, :] <= RCS, axis=-1), cmap=cmap, norm=norm
    )
    ax.invert_yaxis()
    ax.set_title(
        f"Zones for PCL track init for RCS = {RCS} $m^2$ (sensors {[s.id for s in example_sensors]})"
    )

    ax.set_xlabel("lon [°]")
    ax.set_ylabel("lat [°]")
    ax.set_xticks(np.arange(len(lons))[::label_skip])
    ax.set_yticks(np.arange(len(lats))[::label_skip])
    ax.set_xticklabels([f"{lon:.3f}" for lon in lons][::label_skip])
    ax.set_yticklabels([f"{lat:.3f}" for lat in lats][::label_skip])
    ax.tick_params(axis="x", rotation=90)
    fig.savefig(path2, transparent=True)


def setup(app):
    app.connect("builder-inited", save_pcl_example_setup)
    app.connect("builder-inited", save_pcl_coverage_plots)
    
