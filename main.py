import numpy as np
# ---Plotly configuration ---------------------
import plotly.io as pio
pio.renderers.default = 'browser'
# ---------------------------------------------

# Import Hypercube from its structure module
from src.hypercube.structure import Hypercube
# Import the plotting function from the new visualize module
from src.hypercube.vizualize import plot_hypercube_3d


def main():
    print("--- 3D Cube Example ---")
    cube = Hypercube(3)
    print(f"Cube dimension: {cube.dimension}")

    # Plot initial 3D cube
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        title="Initial 3D Cube"
    )

    print("\n--- Testing Translation ---")
    cube.translate([-0.5, -0.5, -0.5]) # Center the cube around origin for better rotation visualization
    print("Translated cube vertex 0 (centered):", cube.vertices_map[0].coordinates.tolist())
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        title="3D Cube after Centering Translation"
    )

    # Test rotation (rotate around Z-axis, which is plane_indices [0,1] or X-Y plane)
    print("\n--- Testing Rotation (around XY-plane by 45 degrees) ---")
    cube.rotate([0, 1], np.pi / 4) # Rotate X-Y plane by 45 degrees
    print("Rotated cube vertex 0:", cube.vertices_map[0].coordinates.tolist())
    plot_hypercube_3d(
        vertices_dict=cube.vertices_map,
        edges_list=cube.get_edges_as_pairs(),
        title="3D Cube after XY Rotation"
    )

    # --- 4D Tesseract Examples ---
    tesseract = Hypercube(4)
    print("\n--- 4D Tesseract Example ---")
    print(f"Tesseract dimension: {tesseract.dimension}")

    # Test orthogonal projection (4D to 3D)
    print("\n--- Testing Orthogonal Projection (4D to 3D) ---")
    projected_3d_ortho = tesseract.project(3, projection_type='orthogonal')
    plot_hypercube_3d(
        vertices_dict=projected_3d_ortho,
        edges_list=tesseract.get_edges_as_pairs(), # Use original edges as topology is preserved
        title="4D Tesseract: Orthogonal Projection to 3D"
    )

    # Test perspective projection (4D to 3D)
    print("\n--- Testing Perspective Projection (4D to 3D, view_distance=3.0) ---")
    projected_3d_persp = tesseract.project(3, projection_type='perspective', perspective_params={'view_distance': 3.0})
    plot_hypercube_3d(
        vertices_dict=projected_3d_persp,
        edges_list=tesseract.get_edges_as_pairs(),
        title="4D Tesseract: Perspective Projection to 3D (d=3.0)"
    )

    # Let's try to simulate a W-axis rotation on the tesseract before projecting
    print("\n--- Tesseract: Rotate XW-plane, then Orthogonal Project to 3D ---")
    tesseract_rotated = Hypercube(4) # Create a new tesseract for this test
    tesseract_rotated.translate([-0.5, -0.5, -0.5, -0.5]) # Center it
    tesseract_rotated.rotate([0, 3], np.pi / 4) # Rotate in the XW plane
    projected_3d_rotated_ortho = tesseract_rotated.project(3, projection_type='orthogonal')
    plot_hypercube_3d(
        vertices_dict=projected_3d_rotated_ortho,
        edges_list=tesseract_rotated.get_edges_as_pairs(),
        title="4D Tesseract: XW-Rotation then Orthogonal Projection to 3D"
    )

    print("\n--- Tesseract: Rotate YW-plane, then Perspective Project to 3D ---")
    tesseract_rotated_yw = Hypercube(4)
    tesseract_rotated_yw.translate([-0.5, -0.5, -0.5, -0.5]) # Center it
    tesseract_rotated_yw.rotate([1, 3], np.pi / 2) # Rotate in the YW plane by 90 degrees
    projected_3d_rotated_persp = tesseract_rotated_yw.project(3, projection_type='perspective', perspective_params={'view_distance': 3.0})
    plot_hypercube_3d(
        vertices_dict=projected_3d_rotated_persp,
        edges_list=tesseract_rotated_yw.get_edges_as_pairs(),
        title="4D Tesseract: YW-Rotation then Perspective Projection to 3D"
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

