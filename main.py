import numpy as np
from src.hypercube.structure import Hypercube

def run():
    print("--- 3D Cube Example ---")
    cube = Hypercube(3)
    print(f"Cube dimension: {cube.dimension}")

    print("\nVertices:")
    for idx, v in cube.vertices_map.items():
        print(f"  {idx}: {v.coordinates.tolist()}")

    print("\nEdges (initial_indices):")
    for edge in cube.get_edges_as_pairs():
        print(f"  {edge}")

    print("\nFaces (initial_indices sets):")
    for face in cube.get_faces_as_vertex_indices():
        print(f"  {sorted(list(face))}")

    # Test translation
    print("\n--- Testing Translation ---")
    cube.translate([0.5, 0.5, 0.5])
    print("Translated cube vertex 0:", cube.vertices_map[0].coordinates.tolist())
    print("Translated cube vertex 7:", cube.vertices_map[7].coordinates.tolist())

    # Test rotation (rotate around Z-axis, which is plane_indices [0,1] or X-Y plane)
    print("\n--- Testing Rotation (around XY-plane by 45 degrees) ---")
    cube.rotate([0, 1], np.pi / 4) # Rotate X-Y plane by 45 degrees
    print("Rotated cube vertex 0:", cube.vertices_map[0].coordinates.tolist())
    print("Rotated cube vertex 7:", cube.vertices_map[7].coordinates.tolist())

    # Reset cube for projection test
    cube_for_proj = Hypercube(3)

    # Test orthogonal projection (3D to 2D)
    print("\n--- Testing Orthogonal Projection (3D to 2D) ---")
    projected_2d_ortho = cube_for_proj.project(2, projection_type='orthogonal')
    print("Projected vertex 0 (orthogonal):", projected_2d_ortho[0].coordinates.tolist())
    print("Projected vertex 7 (orthogonal):", projected_2d_ortho[7].coordinates.tolist())

    # Test perspective projection (3D to 2D)
    print("\n--- Testing Perspective Projection (3D to 2D, view_distance=2.0) ---")
    projected_2d_persp = cube_for_proj.project(2, projection_type='perspective', perspective_params={'view_distance': 2.0})
    print("Projected vertex 0 (perspective):", projected_2d_persp[0].coordinates.tolist())
    print("Projected vertex 7 (perspective):", projected_2d_persp[7].coordinates.tolist())


    print("\n--- 4D Tesseract Example ---")
    tesseract = Hypercube(4)
    print(f"Tesseract dimension: {tesseract.dimension}")

    print(f"\nNumber of 0-faces (vertices): {len(tesseract.get_k_faces(0))}")
    print(f"Number of 1-faces (edges): {len(tesseract.get_k_faces(1))}")
    print(f"Number of 2-faces (squares): {len(tesseract.get_k_faces(2))}")
    print(f"Number of 3-faces (cells/cubes): {len(tesseract.get_k_faces(3))}")
    print(f"Number of 4-faces (the tesseract itself): {len(tesseract.get_k_faces(4))}")

    # Test projection of tesseract (4D to 3D)
    print("\n--- Testing Orthogonal Projection (4D to 3D) ---")
    projected_3d_ortho = tesseract.project(3, projection_type='orthogonal')
    print("Projected tesseract vertex 0:", projected_3d_ortho[0].coordinates.tolist())
    print("Projected tesseract vertex 15:", projected_3d_ortho[15].coordinates.tolist()) # (1,1,1,1) -> (1,1,1)

    print("\n--- Testing Perspective Projection (4D to 3D, view_distance=3.0) ---")
    projected_3d_persp = tesseract.project(3, projection_type='perspective', perspective_params={'view_distance': 3.0})
    print("Projected tesseract vertex 0:", projected_3d_persp[0].coordinates.tolist())
    print("Projected tesseract vertex 15:", projected_3d_persp[15].coordinates.tolist())


    print("\n--- Testing Conceptual Unfolding (3D cube) ---")
    unfolded_cube_data = cube_for_proj.unfold(unfolding_type='net_3d_cube', step=0.5)
    if unfolded_cube_data:
        print(f"Unfolded 2D faces of cube (example for one face): {unfolded_cube_data[2][0]}")

    print("\n--- Testing Conceptual Unfolding (4D tesseract) ---")
    unfolded_tesseract_data = tesseract.unfold(unfolding_type='cross_tesseract', step=0.25)
    if unfolded_tesseract_data:
        print(f"Unfolded 3D cells of tesseract (example for one cell): {unfolded_tesseract_data[3][0]}")

if __name__ == "__main__":
    run()

