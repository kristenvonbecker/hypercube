# test_scad.py

from src.hypercube.geometry.structure import Hypercube
from src.hypercube.scad.scad import ScadExporter

import numpy as np

import os
OUTPUT_DIR = "output/scad/tests"


def test_projection(
        projection_type="perspective",
        perspective_params={'view_distance': 1.5}
):
    """
    Tests exporting a 3D perspective projection of a rotated 4D hypercube.
    """
    print("\n--- Generating Test 1: 4D Perspective Projection ---")
    output_file = os.path.join(OUTPUT_DIR, "test_4d_projection.scad")

    # 1. Create and orient the hypercube
    h_cube = Hypercube(4)
    h_cube.translate([-0.5] * 4).rotate(plane_indices=(0, 3), angle=np.pi / 6)

    # 2. Initialize the exporter with the hypercube object
    exporter = ScadExporter(h_cube)

    # 3. Export the desired projection
    exporter.export_projection(
        output_filename=output_file,
        projection_type=projection_type,
        perspective_params=perspective_params
    )


def test_cross_section(slice_value=0.25):
    """
    Tests exporting a 3D cross-section of a doubly-rotated 4D hypercube.
    """
    print("\n--- Generating Test 2: 4D Rotated Cross-Section ---")
    output_file = os.path.join(OUTPUT_DIR, "test_4d_cross_section.scad")

    # 1. Create and orient the hypercube using chained methods
    h_cube = Hypercube(4)
    h_cube.translate([-0.5] * 4) \
        .rotate(plane_indices=(2, 3), angle=np.pi / 4) \
        .rotate(plane_indices=(0, 1), angle=np.pi / 4) \
        .rotate(plane_indices=(1,3), angle=np.pi / 3)

    # 2. Initialize the exporter
    exporter = ScadExporter(h_cube)

    # 3. Export the cross-section
    exporter.export_cross_section(
        output_filename=output_file,
        slice_value=slice_value,
        slice_dimension=3
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("===================================")
    print("= Running SCAD Generation Tests   =")
    print("===================================")

    test_projection()
    test_cross_section()

    print("\n===================================")
    print(f"= Tests complete. Files saved in '{OUTPUT_DIR}'.")
    print("===================================")
