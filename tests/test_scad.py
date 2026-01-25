"""Tests for OpenSCAD export helpers."""

import numpy as np

from solid2 import sphere

from src.hypercube_factory.geometry.structure import Hypercube
from src.hypercube_factory.scad.scad import (
    ScadExporter,
    _save_to_scad,
    _generate_projection_scad_object,
    _generate_cross_section_scad_object,
    _create_cylinder_between_points,
)


def test_save_to_scad_writes_header_and_body(tmp_path):
    scad_obj = sphere(r=1)
    output_path = tmp_path / "simple.scad"

    _save_to_scad(output_path, scad_obj)

    text = output_path.read_text()
    assert text.startswith("$fn = 50;\n")
    assert "sphere" in text


def test_generate_projection_scad_object_contains_primitives():
    hypercube = Hypercube(4)
    scad_obj = _generate_projection_scad_object(
        hypercube, target_dim=3, projection_radius=0.1, projection_type="orthogonal"
    )

    scad_text = scad_obj.as_scad()
    assert "sphere" in scad_text
    assert "cylinder" in scad_text


def test_generate_cross_section_scad_object_with_faces():
    hypercube = Hypercube(3)
    scad_obj = _generate_cross_section_scad_object(
        hypercube, slice_dimension=2, slice_value=0.5
    )

    scad_text = scad_obj.as_scad()
    assert "polyhedron" in scad_text


def test_generate_cross_section_scad_object_without_faces():
    class DummyHypercube:
        dimension = 3

        def cross_section(self, slice_dimension, slice_value):
            return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), None

    scad_obj = _generate_cross_section_scad_object(
        DummyHypercube(), slice_dimension=2, slice_value=0.0
    )

    scad_text = scad_obj.as_scad()
    assert "sphere" in scad_text


def test_create_cylinder_between_points_zero_length_returns_sphere():
    scad_obj = _create_cylinder_between_points([0, 0, 0], [0, 0, 0], radius=0.2)

    scad_text = scad_obj.as_scad()
    assert "sphere" in scad_text


def test_create_cylinder_between_points_nonzero_returns_cylinder():
    scad_obj = _create_cylinder_between_points([0, 0, 0], [0, 0, 1], radius=0.2)

    scad_text = scad_obj.as_scad()
    assert "cylinder" in scad_text


def test_scad_exporter_writes_projection_and_cross_section(tmp_path):
    hypercube = Hypercube(4)
    exporter = ScadExporter(hypercube)

    projection_path = tmp_path / "projection.scad"
    exporter.export_projection(str(projection_path))

    cross_section_path = tmp_path / "cross_section.scad"
    exporter.export_cross_section(str(cross_section_path), slice_value=0.5, slice_dimension=3)

    assert projection_path.exists()
    assert projection_path.read_text().startswith("$fn = 50;\n")
    assert cross_section_path.exists()
    assert cross_section_path.read_text().startswith("$fn = 50;\n")
