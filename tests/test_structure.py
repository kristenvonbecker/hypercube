# test_structure.py

import pytest
import numpy as np
import math

from src.hypercube.geometry.structure import Hypercube

# --- Test Data ---

# A range of dimensions to test.
DIMENSIONS_TO_TEST = [0, 1, 2, 3, 4, 5, 6]


# --- Helper Function for k-face count ---

def get_expected_k_face_count(n, k):
    """Calculates the expected number of k-faces in an n-dimensional hypercube."""
    if k < 0 or k > n:
        return 0
    # The number of k-faces in an n-cube is 2^(n-k) * C(n, k)
    return (2 ** (n - k)) * math.comb(n, k)


# --- Test Cases ---

def test_invalid_instantiation():
    """Tests that Hypercube raises ValueError for invalid dimensions."""
    with pytest.raises(ValueError, match="Dimension must be a non-negative integer."):
        Hypercube(-1)
    with pytest.raises(ValueError, match="Dimension must be a non-negative integer."):
        Hypercube(1.5)
    with pytest.raises(TypeError):  # Note: a string will likely raise a TypeError inside itertools
        Hypercube("four")


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_vertex_generation(dim):
    """
    Tests that the correct number of vertices and 0-faces are generated
    for a given dimension.
    """
    h = Hypercube(dim)
    expected_vertex_count = 2 ** dim

    # Check number of vertices in the map
    assert len(h.vertices_map) == expected_vertex_count, f"Dimension {dim}: Incorrect vertex count."

    # Check that each vertex has the correct coordinate dimension
    if dim > 0:
        for v in h.vertices_map.values():
            assert len(v.coordinates) == dim, f"Dimension {dim}: Incorrect coordinate dimension."

    # Check that 0-faces (vertices) are correctly stored in k_faces
    assert len(h.k_faces.get(0, set())) == expected_vertex_count, f"Dimension {dim}: Incorrect 0-face count."


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_k_face_generation(dim):
    """
    Tests that the correct number of k-faces are generated for k > 0.
    """
    h = Hypercube(dim)

    # Check face counts for all k from 1 to dim
    for k in range(1, dim + 1):
        expected_count = get_expected_k_face_count(dim, k)
        actual_count = len(h.get_k_faces(k))
        assert actual_count == expected_count, f"Dimension {dim}, k={k}: Expected {expected_count} faces, got {actual_count}."

    # Check that there are no faces of dimension > dim
    assert len(h.get_k_faces(dim + 1)) == 0, f"Dimension {dim}: Should be no {dim + 1}-faces."


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_translation(dim):
    """Tests the translate method."""
    h = Hypercube(dim)

    if dim == 0:
        # Translation of a 0D point (in 0D space) is tricky. The vector should be empty.
        translation_vector = []
        h.translate(translation_vector)
        assert np.array_equal(h.vertices_map[0].coordinates, np.array([]))
    else:
        # Standard translation test
        initial_coords = h.get_vertex_coordinates()[0].copy()
        translation_vector = np.ones(dim)
        h.translate(translation_vector)
        assert np.array_equal(h.get_vertex_coordinates()[0], initial_coords + translation_vector)

    # Test for incorrect vector dimension
    with pytest.raises(ValueError):
        invalid_vector = np.ones(dim + 1)
        h.translate(invalid_vector)


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_rotation(dim):
    """Tests the rotate method, especially for dimensions where it's not applicable."""
    h = Hypercube(dim)

    if dim < 2:
        # Rotation is not well-defined for dimensions < 2.
        # The check for plane_indices should raise an error.
        with pytest.raises(ValueError):
            h.rotate([0, 0], np.pi / 2)  # Invalid plane indices
        with pytest.raises(ValueError):
            h.rotate([0, 1], np.pi / 2)  # Indices out of bounds for dim < 2
    else:
        # A valid rotation should not raise an error
        initial_coords = h.get_vertex_coordinates()[1].copy()
        h.rotate([0, 1], np.pi / 2)
        rotated_coords = h.get_vertex_coordinates()[1]
        assert not np.array_equal(initial_coords, rotated_coords)


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_projection(dim):
    """Tests the project method for both valid and invalid target dimensions."""
    h = Hypercube(dim)

    if dim > 0:
        target_dim = dim - 1
        # Orthogonal projection should work
        projected_vertices = h.project(target_dimension=target_dim, projection_type='orthogonal')
        assert len(projected_vertices) == 2 ** dim
        assert len(projected_vertices[0].coordinates) == target_dim

        # Perspective projection should also work
        projected_vertices_persp = h.project(
            target_dimension=target_dim,
            projection_type='perspective',
            perspective_params={'view_distance': 5.0}
        )
        assert len(projected_vertices_persp) == 2 ** dim
        assert len(projected_vertices_persp[0].coordinates) == target_dim

    # Test for invalid target dimensions
    with pytest.raises(ValueError):
        h.project(target_dimension=dim)  # Target must be less than current
    with pytest.raises(ValueError):
        h.project(target_dimension=-1)  # Target must be non-negative


@pytest.mark.parametrize("dim", DIMENSIONS_TO_TEST)
def test_cross_section(dim, capsys):
    """
    Tests the cross_section method.
    `capsys` is a pytest fixture to capture stdout.
    """
    h = Hypercube(dim)
    # Center the hypercube at origin for predictable cross-sections
    h.translate([-0.5] * dim if dim > 0 else [])

    if dim < 2:
        # Should print a message and return None, None
        verts, faces = h.cross_section(hyperplane_value=0.0)
        captured = capsys.readouterr()
        assert "meaningful only for dimensions 2 or higher" in captured.out
        assert verts is None and faces is None
    else:
        # A central slice should produce a valid (n-1)-cube cross-section
        verts, faces = h.cross_section(hyperplane_value=0.0)
        assert verts is not None
        # For a central slice, the number of vertices should be 2^(dim-1)
        # This assumes the convex hull finds all vertices of the (dim-1) cross-section.
        assert len(verts) == 2 ** (dim - 1)
        assert faces is not None

        # A slice outside the cube should yield nothing
        verts_outside, faces_outside = h.cross_section(hyperplane_value=10.0)
        captured = capsys.readouterr()
        assert "No intersection points found" in captured.out
        assert verts_outside is None and faces_outside is None
