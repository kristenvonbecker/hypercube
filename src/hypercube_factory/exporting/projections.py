import os
import numpy as np

from src.hypercube_factory.exporting.filesystem import clear_matching_files
from src.hypercube_factory.exporting.rotations import (
    DEFAULT_ORIENTATION,
    ROTATION_PLANES,
    parse_rotation_planes,
)
from src.hypercube_factory.geometry.structure import Hypercube
from src.hypercube_factory.scad.scad import ScadExporter


DEFAULT_OUTPUT_DIR = "artifacts/scad/projections"


def build_rotated_tesseract(
    init_orientation=None,
    animated_rotation_planes=None,
    animated_rotation_degrees=0.0,
    center=True,
):
    tesseract = Hypercube(4)

    if center:
        tesseract.translate([-0.5, -0.5, -0.5, -0.5])

    if init_orientation is None:
        init_orientation = DEFAULT_ORIENTATION

    for plane_name, plane_indices in ROTATION_PLANES.items():
        angle_degrees = init_orientation.get(plane_name, 0.0)
        if np.isclose(angle_degrees, 0.0):
            continue

        tesseract.rotate(
            plane_indices=plane_indices,
            angle=np.deg2rad(angle_degrees),
        )

    if animated_rotation_planes:
        for plane_indices in animated_rotation_planes:
            if np.isclose(animated_rotation_degrees, 0.0):
                continue

            tesseract.rotate(
                plane_indices=plane_indices,
                angle=np.deg2rad(animated_rotation_degrees),
            )

    return tesseract


def make_projection_set(
    projection_count=16,
    projection_type="perspective",
    perspective_distance=1.5,
    rotation_plane_names=None,
    total_rotation_degrees=90.0,
    init_orientation=None,
    projection_radius=0.1,
    output_dir=DEFAULT_OUTPUT_DIR,
    center=True,
):
    """Exports a sequence of projected tesseract wireframes as SCAD files."""
    if projection_count < 1:
        raise ValueError("projection_count must be at least 1.")

    if projection_type not in {"orthogonal", "perspective"}:
        raise ValueError("projection_type must be 'orthogonal' or 'perspective'.")

    if perspective_distance <= 0:
        raise ValueError("perspective_distance must be positive.")

    if rotation_plane_names is None:
        rotation_plane_names = ["zw"]

    animated_rotation_planes = parse_rotation_planes(rotation_plane_names)

    clear_matching_files(output_dir, pattern="projection_*.scad")

    angles = np.linspace(
        0.0,
        total_rotation_degrees,
        projection_count,
        endpoint=False,
    )

    perspective_params = (
        {"view_distance": perspective_distance}
        if projection_type == "perspective"
        else None
    )

    for frame_index, angle_degrees in enumerate(angles):
        tesseract = build_rotated_tesseract(
            init_orientation=init_orientation,
            animated_rotation_planes=animated_rotation_planes,
            animated_rotation_degrees=float(angle_degrees),
            center=center,
        )

        exporter = ScadExporter(tesseract)

        exporter.export_projection(
            output_filename=os.path.join(
                output_dir,
                f"projection_{frame_index:02d}.scad",
            ),
            projection_radius=projection_radius,
            target_projection_dimension=3,
            projection_type=projection_type,
            perspective_params=perspective_params,
        )
