import numpy as np
from solid2 import sphere, cylinder, polyhedron, union, color
from src.hypercube.geometry.structure import Hypercube


class ScadExporter:
    """
    Handles the export of a Hypercube object to an OpenSCAD file.
    """

    def __init__(self, hypercube_obj: Hypercube):
        self.hypercube = hypercube_obj

    def export_projection(self, output_filename: str,
                          projection_radius: float = 0.05,
                          target_projection_dimension: int = 3,
                          projection_type: str = 'orthogonal',
                          perspective_params: dict = None):
        """
        Generates a .scad file for a wireframe projection of the hypercube.
        """
        print(f"Exporting projection to {output_filename}...")
        scad_object = _generate_projection_scad_object(
            self.hypercube, target_projection_dimension,
            projection_radius, projection_type, perspective_params
        )
        _save_to_scad(output_filename, scad_object)

    def export_cross_section(self, output_filename: str,
                             slice_value: float = 0.0,
                             slice_dimension: int = None):
        """
        Generates a .scad file for a solid cross-section of the hypercube.
        """
        print(f"Exporting cross-section to {output_filename}...")
        scad_object = _generate_cross_section_scad_object(
            self.hypercube, slice_dimension, slice_value
        )
        _save_to_scad(output_filename, scad_object)


# --- Helper Functions ---

def _save_to_scad(output_filename: str, scad_object):
    """Writes the final SCAD object to a file with a header."""
    header = "$fn = 50;\n"
    # CORRECTED: Used .as_scad() instead of .to_scad()
    with open(output_filename, "w") as f:
        f.write(header + scad_object.as_scad())
    print(f"Successfully generated {output_filename}")


def _generate_projection_scad_object(hypercube_obj: Hypercube, target_dim: int, projection_radius: float,
                                     projection_type='orthogonal', perspective_params=None):
    """Generates a solid2 object for a single hypercube projection."""
    projected_vertices_map = hypercube_obj.project(target_dim, projection_type, perspective_params)
    edges = hypercube_obj.get_edges_as_pairs()
    scad_objects = []
    for _, vertex_obj in projected_vertices_map.items():
        scad_objects.append(sphere(r=projection_radius).translate(vertex_obj.coordinates))
    for v_idx1, v_idx2 in edges:
        p1 = projected_vertices_map[v_idx1].coordinates
        p2 = projected_vertices_map[v_idx2].coordinates
        scad_objects.append(_create_cylinder_between_points(p1, p2, projection_radius))
    return union()(*scad_objects)


def _generate_cross_section_scad_object(hypercube_obj: Hypercube, slice_dimension: int, slice_value: float):
    """Generates a solid2 object for a single hypercube cross-section."""
    if slice_dimension is None:
        slice_dimension = hypercube_obj.dimension - 1

    vertices, faces = hypercube_obj.cross_section(slice_dimension, slice_value)

    if vertices is None or len(vertices) == 0:
        print("Warning: No valid cross-section generated.")
        return union()
    if faces is None or len(faces) == 0:
        print("Warning: Cross-section resulted in points/lines, not faces.")
        return union()(*[sphere(r=0.05 * 2).translate(v) for v in vertices])

    return color([0.8, 0.4, 0.2, 1.0])(polyhedron(points=vertices.tolist(), faces=faces.tolist()))


def _create_cylinder_between_points(p1, p2, radius):
    """Creates a cylinder between two points."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2 - p1
    height = np.linalg.norm(vector)

    if np.isclose(height, 0):
        return sphere(r=radius).translate(p1)

    cyl = cylinder(r=radius, h=height, center=False)
    z_axis = np.array([0, 0, 1])
    target_vec = vector / height
    dot_product = np.dot(z_axis, target_vec)

    if np.isclose(dot_product, 1.0):
        return cyl.translate(p1)
    if np.isclose(dot_product, -1.0):
        return cyl.rotate(a=180, v=[0, 1, 0]).translate(p1)

    rot_axis = np.cross(z_axis, target_vec)
    rot_angle = np.degrees(np.arccos(dot_product))
    return cyl.rotate(a=rot_angle, v=rot_axis).translate(p1)
