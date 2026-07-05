from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hypercube_factory.geometry.structure import Hypercube, TesseractNet3D
from src.hypercube_factory.graphics.plot_3d import (
    plot_cross_section_3d,
    plot_hypercube_3d,
    plot_hypercube_net_3d,
)

import numpy as np
# ---Plotly configuration ---------------------
import plotly.io as pio
pio.renderers.default = 'browser'
# ---------------------------------------------


def main():
    print("--- 3D Cube Example ---")
    cube = Hypercube(3)
    print(f"Cube dimension: {cube.dimension}")

    # Plot initial 3D cube
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        faces_list=cube.get_faces_as_vertex_indices(),
        title="Initial 3D Cube"
    )

    print("\n--- Testing Translation ---")
    cube.translate([-0.5, -0.5, -0.5])  # Center the cube around origin for better rotation visualization
    print("Translated cube vertex 0 (centered):", cube.vertices_map[0].coordinates.tolist())
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        faces_list=cube.get_faces_as_vertex_indices(),
        title="3D Cube after Centering Translation"
    )

    # Test rotation (rotate around Z-axis, which is plane_indices [0,1] or X-Y plane)
    print("\n--- Testing Rotation (around XY-plane by 45 degrees) ---")
    cube.rotate([0, 1], np.pi / 4)  # Rotate X-Y plane by 45 degrees
    print("Rotated cube vertex 0:", cube.vertices_map[0].coordinates.tolist())
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        faces_list=cube.get_faces_as_vertex_indices(),
        title="3D Cube after XY Rotation"
    )

    # --- 4D Tesseract Examples ---
    tesseract = Hypercube(4)
    tesseract.translate([-0.5, -0.5, -0.5, -0.5])  # Center for consistent projections
    print("\n--- 4D Tesseract Example ---")
    print(f"Tesseract dimension: {tesseract.dimension}")

    # Test orthogonal projection (4D to 3D)
    print("\n--- Testing Orthogonal Projection (4D to 3D, centered) ---")
    tesseract.rotate([0, 2], np.pi / 6)  # Rotate in XZ
    tesseract.rotate([1, 3], np.pi / 5)  # Rotate in YW
    projected_3d_ortho = tesseract.project(3, projection_type='orthogonal')
    plot_hypercube_3d(
        vertices_dict=projected_3d_ortho,
        edges_list=tesseract.get_edges_as_pairs(),  # Use original edges as topology is preserved
        faces_list=tesseract.get_faces_as_vertex_indices(),
        title="4D Tesseract: Orthogonal Projection to 3D (Centered + Rotated)"
    )

    # Test perspective projection (4D to 3D)
    print("\n--- Testing Perspective Projection (4D to 3D, centered) ---")
    tesseract_persp = Hypercube(4)
    tesseract_persp.translate([-0.5, -0.5, -0.5, -0.5])
    tesseract_persp.rotate([0, 1], np.pi / 4)  # Rotate in XY
    tesseract_persp.rotate([2, 3], np.pi / 6)  # Rotate in ZW
    projected_3d_persp = tesseract_persp.project(
        3,
        projection_type='perspective',
        perspective_params={'view_distance': 3.0}
    )
    plot_hypercube_3d(
        vertices_dict=projected_3d_persp,
        edges_list=tesseract_persp.get_edges_as_pairs(),
        faces_list=tesseract_persp.get_faces_as_vertex_indices(),
        title="4D Tesseract: Perspective Projection to 3D (Centered + Rotated, d=3.0)"
    )

    # Let's try to simulate a W-axis rotation on the tesseract before projecting
    print("\n--- Tesseract: Rotate XW-plane, then Orthogonal Project to 3D ---")
    tesseract_rotated = Hypercube(4)  # Create a new tesseract for this test
    tesseract_rotated.translate([-0.5, -0.5, -0.5, -0.5])  # Center it
    tesseract_rotated.rotate([0, 3], np.pi / 4)  # Rotate in the XW plane
    tesseract_rotated.rotate([1, 2], np.pi / 6)  # Rotate in the YZ plane
    projected_3d_rotated_ortho = tesseract_rotated.project(3, projection_type='orthogonal')
    plot_hypercube_3d(
        vertices_dict=projected_3d_rotated_ortho,
        edges_list=tesseract_rotated.get_edges_as_pairs(),
        faces_list=tesseract_rotated.get_faces_as_vertex_indices(),
        title="4D Tesseract: Double Rotation then Orthogonal Projection to 3D"
    )

    print("\n--- Tesseract: Rotate YW-plane, then Perspective Project to 3D ---")
    tesseract_rotated_yw = Hypercube(4)
    tesseract_rotated_yw.translate([-0.5, -0.5, -0.5, -0.5])  # Center it
    tesseract_rotated_yw.rotate([1, 3], np.pi / 2)  # Rotate in the YW plane by 90 degrees
    tesseract_rotated_yw.rotate([0, 2], np.pi / 5)  # Rotate in the XZ plane
    projected_3d_rotated_persp = tesseract_rotated_yw.project(
        3,
        projection_type='perspective',
        perspective_params={'view_distance': 3.0}
    )
    plot_hypercube_3d(
        vertices_dict=projected_3d_rotated_persp,
        edges_list=tesseract_rotated_yw.get_edges_as_pairs(),
        faces_list=tesseract_rotated_yw.get_faces_as_vertex_indices(),
        title="4D Tesseract: Double Rotation then Perspective Projection to 3D"
    )

    # --- Cross-section examples ---
    print("\n--- Tesseract: Cross-Section (w=0.2, rotated) ---")
    tesseract_section = Hypercube(4)
    tesseract_section.translate([-0.5, -0.5, -0.5, -0.5])
    tesseract_section.rotate([0, 3], np.pi / 5)  # XW
    tesseract_section.rotate([1, 3], np.pi / 4)  # YW
    tesseract_section.rotate([2, 3], np.pi / 6)  # ZW
    tesseract_section.rotate([0, 1], np.pi / 7)  # XY
    section_vertices, section_faces = tesseract_section.cross_section(
        hyperplane_dim=3,
        hyperplane_value=0.2
    )
    plot_cross_section_3d(
        cross_section_vertices=section_vertices,
        cross_section_faces=section_faces,
        title="4D Tesseract: Cross-Section at w=0.2 (Centered + Rotated)"
    )

    print("\n--- Tesseract: Cross-Section (w=-0.1, double-rotated) ---")
    tesseract_section_2 = Hypercube(4)
    tesseract_section_2.translate([-0.5, -0.5, -0.5, -0.5])
    tesseract_section_2.rotate([0, 3], np.pi / 3)  # XW
    tesseract_section_2.rotate([1, 3], np.pi / 5)  # YW
    tesseract_section_2.rotate([2, 3], np.pi / 4)  # ZW
    tesseract_section_2.rotate([0, 2], np.pi / 6)  # XZ
    section_vertices_2, section_faces_2 = tesseract_section_2.cross_section(
        hyperplane_dim=3,
        hyperplane_value=-0.1
    )
    plot_cross_section_3d(
        cross_section_vertices=section_vertices_2,
        cross_section_faces=section_faces_2,
        title="4D Tesseract: Cross-Section at w=-0.1 (Centered + Double Rotated)"
    )

    # --- Net examples (3D polycube nets of the 8 cells) ---
    print("\n--- Tesseract Net: Standard ---")
    net_standard = TesseractNet3D(net_style="standard")
    plot_hypercube_net_3d(
        net_standard,
        title="4D Tesseract Net (Standard)"
    )

    print("\n--- Tesseract Net: Zigzag ---")
    net_zigzag = TesseractNet3D(net_style="zigzag")
    plot_hypercube_net_3d(
        net_zigzag,
        title="4D Tesseract Net (Zigzag)"
    )

    # Unfolding is conceptual, so plotting it directly is harder without
    # defining the 3D coordinates of the unfolded state within the unfold method.
    # For now, we'll just show the print statements.
    # print("\n--- Testing Conceptual Unfolding (3D cube) ---")
    # unfolded_cube_data = cube.unfold(unfolding_type='net_3d_cube', step=0.5)
    # if unfolded_cube_data:
    #     print(f"Unfolded 2D faces of cube (example for one face): {unfolded_cube_data[2][0]}")
    #
    # print("\n--- Testing Conceptual Unfolding (4D tesseract) ---")
    # unfolded_tesseract_data = tesseract.unfold(unfolding_type='cross_tesseract', step=0.25)
    # if unfolded_tesseract_data:
    #     print(f"Unfolded 3D cells of tesseract (example for one cell): {unfolded_tesseract_data[3][0]}")


if __name__ == "__main__":
    main()
