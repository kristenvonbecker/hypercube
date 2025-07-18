import numpy as np
# some of the solid2 imports are incorrect
from solid2 import sphere, cylinder, polyhedron, union, translate, rotate, color
from solid2.core.object_base import module, ref, comment, empty # These are often directly on object_base now
from solid2.core.utils import Code # Code is now typically in solid2.core.utils

import os

from structure import Hypercube, Vertex, Face # Corrected import path

# --- Helper Functions for SCAD Generation with solid2 ---


def _create_sphere(point, radius):
    """Creates a sphere at the given point."""
    return sphere(r=radius).translate(point)


def _create_cylinder_between_points(p1, p2, radius):
    """
    Creates a cylinder between two points using solid2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2 - p1
    height = np.linalg.norm(vector)
    if height == 0:
        # If points are identical, return a sphere at that point
        return sphere(r=radius).translate(p1)

    angle_xy = np.degrees(np.arctan2(vector[1], vector[0]))
    angle_z = np.degrees(np.arccos(vector[2] / height))

    # Apply rotations and translation to align the cylinder
    # solid2's cylinder is along the Z-axis by default
    return cylinder(r=radius, h=height).rotate(a=angle_xy, v=[0, 0, 1]).rotate(a=angle_z, v=[0, 1, 0]).translate(p1)


def _generate_projection_scad_object(hypercube_obj: Hypercube, target_dim: int, projection_radius: float, frame_index: int, total_frames: int, projection_type='orthogonal', perspective_params=None):
    """
    Generates a solid2 OpenSCAD object for a single hypercube projection (wireframe).
    Args:
        hypercube_obj (Hypercube): The Hypercube object to project.
        target_dim (int): The dimension to project onto (e.g., 3).
        projection_radius (float): Radius for spheres (vertices) and cylinders (edges).
        frame_index (int): The index of the current animation frame (for potential grouping).
        total_frames (int): Total number of frames (for potential grouping).
        projection_type (str): 'orthogonal' or 'perspective'.
        perspective_params (dict): Params for perspective projection.
    Returns:
        solid2.core.object_base.OpenSCADObject: A solid2 object representing the wireframe projection.
    """
    projected_vertices_map = hypercube_obj.project(target_dim, projection_type, perspective_params)
    edges = hypercube_obj.get_edges_as_pairs()

    scad_objects = []

    # Pass the actual float value directly for radius.
    for idx, vertex_obj in projected_vertices_map.items():
        scad_objects.append(_create_sphere(vertex_obj.coordinates, projection_radius))

    for v_idx1, v_idx2 in edges:
        p1 = projected_vertices_map[v_idx1].coordinates
        p2 = projected_vertices_map[v_idx2].coordinates
        scad_objects.append(_create_cylinder_between_points(p1, p2, projection_radius))

    return union()(*scad_objects).add_comment(f"Projection Frame {frame_index+1}/{total_frames}")


def _generate_cross_section_scad_object(hypercube_obj: Hypercube, hyperplane_dim: int, hyperplane_value: float, frame_index: int, total_frames: int):
    """
    Generates a solid2 OpenSCAD object for a single hypercube cross-section (solid polyhedron).
    Args:
        hypercube_obj (Hypercube): The Hypercube object to slice.
        hyperplane_dim (int): The dimension along which to slice.
        hyperplane_value (float): The value of the hyperplane.
        frame_index (int): The index of the current animation frame.
        total_frames (int): Total number of frames.
    Returns:
        solid2.core.object_base.OpenSCADObject: A solid2 object representing the solid cross-section.
    """
    cross_section_vertices, cross_section_faces = hypercube_obj.cross_section(hyperplane_dim, hyperplane_value)

    if cross_section_vertices is None or len(cross_section_vertices) == 0:
        print(f"Warning: No valid cross-section generated for frame {frame+1} at hyperplane_value={hyperplane_value}.")
        return empty().add_comment(f"Empty Cross-section Frame {frame_index+1}/{total_frames}")

    if cross_section_faces is None or len(cross_section_faces) == 0:
        print(f"Warning: Cross-section for frame {frame_index+1} at hyperplane_value={hyperplane_value} resulted in points/lines, not faces.")
        # If no faces, assume it's points or lines and render them as spheres.
        # Make radius slightly larger for visibility
        return union()(*[_create_sphere(v, 0.05 * 2) for v in cross_section_vertices]).add_comment(f"Points/Lines Cross-section Frame {frame_index+1}/{total_frames}")


    scad_poly = polyhedron(
        points=[v.tolist() for v in cross_section_vertices],
        faces=cross_section_faces.tolist()
    )
    # Corrected: Use Code("render_color") to refer to the OpenSCAD variable
    return color(Code("render_color"))(scad_poly).add_comment(f"Cross-section Frame {frame_index+1}/{total_frames}")

# --- Main Export Functions ---

def export_projections_to_scad(output_filename: str, hypercube_dimension: int, num_frames: int = 16,
                               projection_radius: float = 0.05, rotation_angle_step: float = np.pi / 8,
                               projection_plane_indices: tuple = (0, 1), # Example for 2D rotation
                               target_projection_dimension: int = 3, # Project to 3D for visualization
                               projection_type: str = 'orthogonal', perspective_params: dict = None):
    """
    Generates an OpenSCAD file containing animated wireframe projections of a hypercube.
    Each frame represents a rotated view.
    """
    print(f"Generating {num_frames} hypercube projections to {output_filename}...")

    # Define global parameters for the SCAD file directly using Code for OpenSCAD variables
    global_params = Code(f"""
$fn = 50; // Facets for curves and spheres
wireframe_radius = {projection_radius}; // Radius for vertices and edges - This variable is now decorative, as Python's float is used.
""")
    all_projection_frames = [
        global_params.add_comment("Global Parameters for Projections")
    ]

    hypercube = Hypercube(hypercube_dimension)
    hypercube.translate([-0.5] * hypercube_dimension)

    for i in range(num_frames):
        current_hypercube = Hypercube(hypercube_dimension)
        current_hypercube.translate([-0.5] * hypercube_dimension)

        current_hypercube.rotate(projection_plane_indices, i * rotation_angle_step)

        frame_scad_object = _generate_projection_scad_object(
            current_hypercube, target_projection_dimension,
            projection_radius, # Corrected: Pass the actual float value here
            i, num_frames,
            projection_type, perspective_params
        )
        all_projection_frames.append(
            module(name=f"projection_frame_{i+1}")(
                frame_scad_object
            ).add_comment(f"// Module for Projection Frame {i+1}")
        )

    # Main render block for OpenSCAD to control animation
    # Define current_frame and render_all_projections directly in OpenSCAD with Code
    render_control_vars = Code(f"""
current_frame = 1; // Change this value to render different frames (1 to {num_frames})
render_all_projections = true; // Set to true to render all frames side-by-side
""")
    all_projection_frames.append(render_control_vars)

    all_translated_frames = union()
    for i in range(num_frames):
        all_translated_frames += ref(f"projection_frame_{i+1}")().translate([ (i % 4) * 2.5, (i // 4) * 2.5, 0 ])

    # Use solid2.Code for the raw OpenSCAD conditional logic
    main_render_logic = Code(f"""
if (render_all_projections) {{
    {all_translated_frames.to_scad()}
}} else {{
    projection_frame_current_frame();
}}
""").add_comment("Main Render Block for Projections")

    all_projection_frames.append(main_render_logic)

    root_object = union()(*all_projection_frames)
    root_object.save_as_scad(output_filename)
    print(f"Successfully generated {output_filename}")


def export_cross_sections_to_scad(output_filename: str, hypercube_dimension: int, num_frames: int = 16,
                                  slice_dimension: int = None, min_slice_value: float = -0.7, max_slice_value: float = 0.7):
    """
    Generates an OpenSCAD file containing animated solid cross-sections of a hypercube.
    Each frame represents a slice at a different position.
    """
    print(f"Generating {num_frames} hypercube cross-sections to {output_filename}...")

    # Define global parameters for the SCAD file
    global_params = Code(f"""
$fn = 50;
render_color = [0.8, 0.4, 0.2, 1.0]; // Opaque orange-ish color
""")
    all_cross_section_frames = [
        global_params.add_comment("Global Parameters for Cross-sections")
    ]

    hypercube = Hypercube(hypercube_dimension)
    hypercube.translate([-0.5] * hypercube_dimension)

    if slice_dimension is None:
        slice_dimension = hypercube_dimension - 1

    slice_values = np.linspace(min_slice_value, max_slice_value, num_frames)

    for i, current_slice_value in enumerate(slice_values):
        frame_scad_object = _generate_cross_section_scad_object(
            hypercube, slice_dimension, current_slice_value, i, num_frames
        )
        all_cross_section_frames.append(
            module(name=f"cross_section_frame_{i+1}")(
                # Corrected: Color is now applied inside _generate_cross_section_scad_object using Code("render_color")
                frame_scad_object
            ).add_comment(f"// Module for Cross-section Frame {i+1}")
        )

    main_render_block_vars = Code(f"""
current_frame = 1; // Change this value to render different frames (1 to {num_frames})
render_all_cross_sections = true; // Set to true to render all frames side-by-side
""")
    all_cross_section_frames.append(main_render_block_vars)

    all_translated_frames = union()
    for i in range(num_frames):
        all_translated_frames += ref(f"cross_section_frame_{i+1}")().translate([ (i % 4) * 2.5, (i // 4) * 2.5, 0 ])

    main_render_logic = Code(f"""
if (render_all_cross_sections) {{
    {all_translated_frames.to_scad()}
}} else {{
    cross_section_frame_current_frame();
}}
""").add_comment("Main Render Block for Cross-sections")

    all_cross_section_frames.append(main_render_logic)

    root_object = union()(*all_cross_section_frames)
    root_object.save_as_scad(output_filename)
    print(f"Successfully generated {output_filename}")