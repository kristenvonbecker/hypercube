import os
import numpy as np

from src.hypercube_factory.exporting.filesystem import clear_matching_files
from src.hypercube_factory.exporting.rotations import (
    DEFAULT_ORIENTATION,
    ROTATION_PLANES,
)
from src.hypercube_factory.geometry.structure import Hypercube
from src.hypercube_factory.scad.scad import ScadExporter


DEFAULT_OUTPUT_DIR = "artifacts/scad/slices"


def build_oriented_tesseract(orientation=None):
    tesseract = Hypercube(4)
    tesseract.translate([-0.5, -0.5, -0.5, -0.5])

    if orientation is None:
        orientation = DEFAULT_ORIENTATION

    for plane_name, plane_indices in ROTATION_PLANES.items():
        angle_degrees = orientation[plane_name]
        if np.isclose(angle_degrees, 0.0):
            continue

        tesseract.rotate(
            plane_indices=plane_indices,
            angle=np.deg2rad(angle_degrees),
        )

    return tesseract


def get_slice_bounds(hypercube, slice_dimension=3, padding=0.0):
    vertex_coordinates = hypercube.get_vertex_coordinates().values()
    axis_values = np.array(
        [coordinates[slice_dimension] for coordinates in vertex_coordinates],
        dtype=float,
    )

    slice_min = float(axis_values.min())
    slice_max = float(axis_values.max())

    return slice_min + padding, slice_max - padding


def make_sliceset(
    section_count=16,
    slice_min=None,
    slice_max=None,
    orientation=None,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    clear_matching_files(output_dir, pattern="slice_*.scad")

    if section_count < 1:
        raise ValueError("section_count must be at least 1.")

    oriented_tesseract = build_oriented_tesseract(orientation=orientation)
    computed_slice_min, computed_slice_max = get_slice_bounds(
        oriented_tesseract,
        slice_dimension=3,
    )

    if slice_min is None:
        slice_min = computed_slice_min
    if slice_max is None:
        slice_max = computed_slice_max

    slice_values = np.linspace(slice_min, slice_max, section_count)

    for i, slice_value in enumerate(slice_values):
        tesseract = build_oriented_tesseract(orientation=orientation)

        exporter = ScadExporter(tesseract)

        exporter.export_cross_section(
            output_filename=os.path.join(output_dir, f"slice_{i:02d}.scad"),
            slice_value=float(slice_value),
            slice_dimension=3,
        )
