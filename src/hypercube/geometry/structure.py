import numpy as np
import itertools
from itertools import combinations
from scipy.spatial import ConvexHull  # New import for cross_section
from collections import defaultdict  # New import for cross_section


# --- Vertex Class ---
class Vertex:
    """
    Represents a single vertex of the hypercube.
    Attributes:
        coordinates (np.array): The current N-dimensional coordinates of the vertex.
        initial_index (int): The original index of the vertex when the hypercube was created.
        color (str/tuple): Placeholder for visualization attributes (e.g., color).
        highlighted (bool): Placeholder for visualization attributes (e.g., highlighting).
    """

    def __init__(self, coordinates, initial_index=None):
        self.coordinates = np.array(coordinates, dtype=float)
        self.initial_index = initial_index
        self.color = None
        self.highlighted = False

    def __repr__(self):
        """String representation of the Vertex object."""
        return f"Vertex(id={self.initial_index}, coords={self.coordinates.tolist()})"

    def __hash__(self):
        """Enables hashing for use in sets/dictionaries, primarily using initial_index."""
        return hash(self.initial_index) if self.initial_index is not None else hash(tuple(self.coordinates.tolist()))

    def __eq__(self, other):
        """Defines equality, primarily using initial_index."""
        if not isinstance(other, Vertex):
            return NotImplemented
        if self.initial_index is not None and other.initial_index is not None:
            return self.initial_index == other.initial_index
        # Fallback if initial_index is not set for some reason
        return np.array_equal(self.coordinates, other.coordinates)


# --- Face Class ---
class Face:
    """
    Represents a k-dimensional face of the hypercube (e.g., vertex (0D), edge (1D), square (2D), cube (3D)).
    Attributes:
        vertices_indices (frozenset): A frozenset of initial_indices of the vertices that define this face.
                                      Using frozenset makes the Face object hashable.
        dimension (int): The dimensionality of this face (e.g., 0 for a vertex, 1 for an edge).
        _all_vertices_map (dict): A reference to the parent Hypercube's vertices_map,
                                  allowing access to actual Vertex objects.
    """

    def __init__(self, vertices_indices, dimension, all_vertices_map):
        self.vertices_indices = frozenset(vertices_indices)
        self.dimension = dimension
        self._all_vertices_map = all_vertices_map

    @property
    def vertices(self):
        """Returns a list of Vertex objects corresponding to the face's vertices_indices."""
        return [self._all_vertices_map[idx] for idx in self.vertices_indices]

    def __repr__(self):
        """String representation of the Face object."""
        return f"Face(dim={self.dimension}, indices={sorted(list(self.vertices_indices))})"

    def __hash__(self):
        """Enables hashing for use in sets/dictionaries."""
        return hash((self.dimension, self.vertices_indices))

    def __eq__(self, other):
        """Defines equality for Face objects."""
        return isinstance(other, Face) and \
            self.dimension == other.dimension and \
            self.vertices_indices == other.vertices_indices


# --- Hypercube Class ---
class Hypercube:
    """
    Represents an N-dimensional hypercube.
    Attributes:
        dimension (int): The dimension of the hypercube.
        vertices_map (dict): Maps initial vertex indices to Vertex objects.
        k_faces (dict): Stores sets of k-dimensional Face objects, keyed by dimension.
    """

    def __init__(self, dimension):
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError("Dimension must be a non-negative integer.")
        self.dimension = dimension
        self.vertices_map = {}  # Maps initial_index to Vertex object
        self.k_faces = {}  # Dictionary to store sets of k-dimensional faces

        self._generate_vertices()
        self._generate_k_faces()

    def _generate_vertices(self):
        """Generates all 2^N vertices of the N-dimensional hypercube."""
        interval = [0, 1]  # Vertices are at (0,0,..,0) to (1,1,..,1)
        coordinates = np.array(
            [[val for val in prod] for prod in itertools.product(interval, repeat=self.dimension)]
        )
        for i, coord in enumerate(coordinates):
            self.vertices_map[i] = Vertex(coord, initial_index=i)
        # 0-dimensional faces are just the vertices themselves
        self.k_faces[0] = {Face({i}, 0, self.vertices_map) for i in self.vertices_map}

    def _generate_k_faces(self):
        """
        Generates k-dimensional faces for all k from 1 to self.dimension.
        A k-face is defined by holding (N-k) coordinates fixed and letting k coordinates vary.
        """
        if self.dimension == 0:
            return  # A 0-dimensional hypercube has only 0-faces (vertices)

        for d in range(1, self.dimension + 1):
            current_dim_faces = set()
            # Iterate through all combinations of 'd' dimensions that will vary
            for varying_dims_indices in combinations(range(self.dimension), d):
                fixed_dims_indices = [i for i in range(self.dimension) if i not in varying_dims_indices]

                # Iterate through all 2^(N-d) combinations of fixed values for the fixed dimensions
                for fixed_values_tuple in itertools.product([0, 1], repeat=len(fixed_dims_indices)):
                    face_vertex_indices = set()
                    # For each combination of fixed dimensions/values, generate the 2^d vertices of this face
                    for varying_values_tuple in itertools.product([0, 1], repeat=len(varying_dims_indices)):
                        vertex_coords = [None] * self.dimension
                        v_idx_counter = 0  # Counter for varying_values_tuple
                        f_idx_counter = 0  # Counter for fixed_values_tuple

                        # Reconstruct the full N-dimensional coordinates for this vertex
                        for dim_idx in range(self.dimension):
                            if dim_idx in varying_dims_indices:
                                vertex_coords[dim_idx] = varying_values_tuple[v_idx_counter]
                                v_idx_counter += 1
                            else:
                                vertex_coords[dim_idx] = fixed_values_tuple[f_idx_counter]
                                f_idx_counter += 1

                        # Find the initial_index of this vertex in the global map
                        # This lookup ensures we use the canonical Vertex object for the face
                        for idx, v_obj in self.vertices_map.items():
                            if np.array_equal(v_obj.coordinates, np.array(vertex_coords)):
                                face_vertex_indices.add(idx)
                                break
                    current_dim_faces.add(Face(face_vertex_indices, d, self.vertices_map))
            self.k_faces[d] = current_dim_faces

    def translate(self, vector):
        """
        Translates the entire hypercube by a given vector.
        Args:
            vector (list/np.array): A list or numpy array of length `self.dimension`.
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Translation vector must match hypercube dimension ({self.dimension}).")
        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates += np.array(vector, dtype=float)

    def rotate(self, plane_indices, angle):
        """
        Rotates the hypercube in a specified 2D plane.
        Args:
            plane_indices (list/tuple): A pair of integers [i, j] representing the axes of the rotation plane.
                                        e.g., [0, 1] for XY plane, [0, 3] for XW plane in a 4D hypercube.
            angle (float): The rotation angle in radians.
        """
        if len(plane_indices) != 2 or not all(0 <= i < self.dimension for i in plane_indices):
            raise ValueError(f"Plane indices must be a pair of integers within [0, {self.dimension - 1}].")

        i, j = plane_indices
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        # Create a rotation matrix for the specified plane
        rotation_matrix = np.identity(self.dimension)
        rotation_matrix[i, i] = cos_theta
        rotation_matrix[i, j] = -sin_theta
        rotation_matrix[j, i] = sin_theta
        rotation_matrix[j, j] = cos_theta

        # Apply the rotation matrix to each vertex's coordinates
        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates = np.dot(rotation_matrix, vertex_obj.coordinates)

    def project(self, target_dimension, projection_type='orthogonal', perspective_params=None):
        """
        Projects the hypercube onto a lower-dimensional space.
        Args:
            target_dimension (int): The dimension of the space to project onto (e.g., 3 for 3D visualization).
            projection_type (str): The type of projection: 'orthogonal' or 'perspective'.
            perspective_params (dict, optional): Parameters for perspective projection.
                                                 For 'perspective', must include 'view_distance'.
                                                 e.g., {'view_distance': 5.0}

        Returns:
            dict: A dictionary mapping initial vertex indices to new Vertex objects
                  with their projected coordinates. These new Vertex objects are copies
                  and do not modify the original hypercube.
        """
        if not (0 <= target_dimension < self.dimension):
            raise ValueError(
                f"Target dimension must be less than current dimension ({self.dimension}) and non-negative.")

        projected_vertices = {}

        if projection_type == 'orthogonal':
            # Orthogonal projection: simply truncate coordinates to the target_dimension
            for idx, vertex_obj in self.vertices_map.items():
                projected_coords = vertex_obj.coordinates[:target_dimension]
                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        elif projection_type == 'perspective':
            if perspective_params is None or 'view_distance' not in perspective_params:
                raise ValueError("Perspective projection requires 'view_distance' in perspective_params.")

            view_distance = float(perspective_params['view_distance'])
            if view_distance <= 0:
                raise ValueError("View distance for perspective projection must be positive.")

            # Simple perspective projection (e.g., projecting from N-D to N-1D
            # by treating the last dimension as a 'depth' axis).
            # (x_0, ..., x_{N-2}, x_{N-1}) -> (x_0_p, ..., x_{N-2}_p)
            # where x_i_p = x_i * view_distance / (view_distance - x_{N-1})

            # This handles projection from self.dimension to self.dimension - 1.
            # If target_dimension < self.dimension - 1, it first projects to self.dimension - 1
            # and then truncates the result. A more general N-D to M-D perspective projection
            # requires a custom projection matrix that would be passed in.

            for idx, vertex_obj in self.vertices_map.items():
                original_coords = vertex_obj.coordinates.copy()
                projected_coords = np.zeros(target_dimension)

                # Determine the 'depth' coordinate for perspective effect
                # We'll use the last dimension of the original hypercube for perspective division
                if self.dimension > 0:  # Ensure there's a last coordinate to use
                    depth_coord = original_coords[self.dimension - 1]
                else:  # For 0D hypercube (a point), no perspective applies
                    depth_coord = 0.0

                divisor = (view_distance - depth_coord)

                if divisor == 0:
                    # Handle division by zero (points on the projection plane)
                    # For visualization, it's often better to clip these or place them far away
                    # Here, we use a tiny non-zero value to prevent errors.
                    divisor = 1e-9  # A very small epsilon to avoid division by zero

                # Apply perspective effect to the first `target_dimension` coordinates
                for i in range(target_dimension):
                    # Only apply perspective if original dimension has more than `i` coordinates
                    if i < self.dimension:
                        projected_coords[i] = original_coords[i] * view_distance / divisor
                    else:
                        # If target_dimension is higher than available original dimensions (should not happen
                        # given checks, but for safety)
                        projected_coords[i] = 0.0

                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}. Choose 'orthogonal' or 'perspective'.")

        return projected_vertices

    # def unfold(self, unfolding_type='default', step=0.0):
    #     """
    #     Conceptual method for unfolding the hypercube.
    #     This method would transform the hypercube's components (vertices of its faces)
    #     to simulate an unfolding process in a lower dimension.
    #
    #     Args:
    #         unfolding_type (str): Specifies which unfolding strategy to use.
    #                               - 'default': A basic conceptual unfolding (e.g., simple rotations).
    #                               - 'net_3d_cube': For 3D cube, unfolds into a 2D cross net.
    #                               - 'cross_tesseract': For 4D hypercube, unfolds into 8 3D cubes in a cross shape.
    #         step (float): A normalized animation parameter from 0.0 (fully folded) to 1.0 (fully unfolded).
    #                       Intermediate values represent partial unfolding.
    #
    #     Returns:
    #         dict: A dictionary containing lists of transformed k-faces.
    #               Each k-face will be represented by a dict with its original indices
    #               and its new, transformed coordinates for visualization.
    #               Returns None if the type is not implemented.
    #     """
    #     if not (0.0 <= step <= 1.0):
    #         raise ValueError("Unfolding step must be between 0.0 and 1.0.")
    #
    #     # Create a copy of vertex coordinates to work with for unfolding
    #     # This allows modifications without affecting the original hypercube state
    #     unfolded_vertex_coords = {
    #         idx: v.coordinates.copy() for idx, v in self.vertices_map.items()
    #     }
    #
    #     # The actual unfolding logic is complex and depends on the specific type
    #     # For a practical zoetrope, you'd likely implement specific cases.
    #
    #     if self.dimension == 3 and unfolding_type == 'net_3d_cube':
    #         # Example: Unfolding a 3D cube into a 2D cross net
    #         # This would involve rotations of faces around shared edges onto a common plane.
    #         # The 'step' would control the angle of rotation.
    #
    #         # Identify the "central" face, e.g., the z=0 face (indices {0, 2, 4, 6})
    #         central_face_indices = frozenset({0, 2, 4, 6})
    #
    #         # Simplified rotation of other faces relative to the central face
    #         # For each face, identify its shared edge with the central face
    #         # and rotate it around that edge.
    #
    #         # This part is illustrative. Actual rotations require finding rotation axes and applying transforms.
    #         # For a real implementation, you'd define which faces unfold from which edge.
    #         print(f"Applying '{unfolding_type}' unfolding for 3D cube at step {step}")
    #
    #     elif self.dimension == 4 and unfolding_type == 'cross_tesseract':
    #         # This is the classic unfolding of a tesseract into eight 3D cubes.
    #         # One 3D cube is central, and the other seven are rotated/translated
    #         # in 3D space to form the cross shape.
    #         # The 'step' would control the rotation angle of these 3D cells.
    #
    #         # This would involve selecting one 'central' 3-face (a 3D cube).
    #         # Then, for each adjacent 3-face, identify their common 2-face (a square)
    #         # and rotate the adjacent 3-face around that common 2-face.
    #         print(f"Applying '{unfolding_type}' unfolding for 4D hypercube at step {step}")
    #
    #     else:
    #         print(
    #             f"Unfolding type '{unfolding_type}' for {self.dimension}-D hypercube is not specifically implemented here.")
    #         print("Returning current coordinates for all vertices/faces.")
    #
    #     # Construct the output format: a dict of lists of transformed k-faces
    #     transformed_k_faces = {}
    #     for k, faces_set in self.k_faces.items():
    #         transformed_k_faces[k] = []
    #         for face_obj in faces_set:
    #             face_coords_list = []
    #             for v_idx in face_obj.vertices_indices:
    #                 # In a full implementation, `unfolded_vertex_coords[v_idx]` would be the
    #                 # transformed coordinate for `v_idx` based on the unfolding logic.
    #                 face_coords_list.append(unfolded_vertex_coords[v_idx].tolist())
    #
    #             # You might also want to add the ordered vertices for faces for rendering
    #             # (This would be another method, e.g., `face_obj.get_ordered_coordinates()`)
    #             transformed_k_faces[k].append({
    #                 'initial_indices': sorted(list(face_obj.vertices_indices)),
    #                 'dimension': face_obj.dimension,
    #                 'current_coords': face_coords_list
    #             })
    #     return transformed_k_faces

    def cross_section(self, hyperplane_dim=None, hyperplane_value=0.0):
        """
        Calculates the cross-section of the hypercube with a given hyperplane.
        Assumes the hyperplane is defined by one dimension fixed at a certain value.
        For example, hyperplane_dim=3, hyperplane_value=0.0 for w=0 in a 4D hypercube.

        Args:
            hyperplane_dim (int): The index of the dimension to slice along (e.g., 3 for 'w' in a 4D hypercube).
                                  If None, defaults to the last dimension (self.dimension - 1).
            hyperplane_value (float): The value at which the hyperplane slices that dimension.

        Returns:
            tuple: (cross_section_vertices, cross_section_faces)
                   - cross_section_vertices (np.array): New vertices of the cross-section,
                                                        with the hyperplane_dim removed.
                   - cross_section_faces (np.array): Indices into cross_section_vertices,
                                                     defining triangular faces of the cross-section.
                                                     Returns None for faces if not applicable (e.g., 1D result).
        """
        if self.dimension < 2:
            print("Cross-section is meaningful only for dimensions 2 or higher.")
            return None, None

        if hyperplane_dim is None:
            hyperplane_dim = self.dimension - 1

        if not (0 <= hyperplane_dim < self.dimension):
            raise ValueError(f"Hyperplane dimension must be within [0, {self.dimension - 1}].")

        intersection_points_full_dim = []
        original_edges = self.get_edges_as_pairs()  # Get edges based on initial_indices

        # Map current vertex initial_indices to their current coordinates
        current_vertex_coords = self.get_vertex_coordinates()

        for idx1, idx2 in original_edges:
            v1_coords = current_vertex_coords[idx1]
            v2_coords = current_vertex_coords[idx2]

            coord1_val = v1_coords[hyperplane_dim]
            coord2_val = v2_coords[hyperplane_dim]

            # Check if the edge crosses the hyperplane
            # Case 1: One endpoint is on the hyperplane
            # Use np.isclose for floating point comparisons
            is_v1_on_hyperplane = np.isclose(coord1_val, hyperplane_value)
            is_v2_on_hyperplane = np.isclose(coord2_val, hyperplane_value)

            if is_v1_on_hyperplane and is_v2_on_hyperplane:
                # Both endpoints are on the hyperplane, the entire edge is part of the cross-section
                # We add both endpoints. Duplicate handling will occur later.
                intersection_points_full_dim.append(v1_coords)
                intersection_points_full_dim.append(v2_coords)
            elif is_v1_on_hyperplane:
                # Only v1 is on the hyperplane
                intersection_points_full_dim.append(v1_coords)
            elif is_v2_on_hyperplane:
                # Only v2 is on the hyperplane
                intersection_points_full_dim.append(v2_coords)
            # Case 2: The hyperplane intersects the interior of the edge
            elif (coord1_val < hyperplane_value and coord2_val > hyperplane_value) or \
                    (coord1_val > hyperplane_value and coord2_val < hyperplane_value):
                # Linear interpolation to find the intersection point
                # t = (hyperplane_value - coord1_val) / (coord2_val - coord1_val)
                # P = v1 + t * (v2 - v1)
                delta = coord2_val - coord1_val
                if np.isclose(delta, 0):  # Should not happen if signs are different, but for safety
                    continue
                t = (hyperplane_value - coord1_val) / delta
                intersection_coord_full_dim = v1_coords + t * (v2_coords - v1_coords)
                intersection_points_full_dim.append(intersection_coord_full_dim)

        if not intersection_points_full_dim:
            print(f"No intersection points found for hyperplane dim {hyperplane_dim} at value {hyperplane_value}.")
            return None, None

        # Convert to numpy array and remove duplicates using rounding for float precision
        # Convert each point to a tuple for hashing in a set, round to avoid floating point issues
        unique_intersection_points_full_dim = np.array(list(
            {tuple(p.round(8)) for p in intersection_points_full_dim}
        ))

        # Project the unique intersection points onto the (N-1)D space by deleting the hyperplane_dim
        cross_section_vertices = np.delete(unique_intersection_points_full_dim, hyperplane_dim, axis=1)

        target_hull_dimension = self.dimension - 1

        if len(cross_section_vertices) < target_hull_dimension + 1:
            print(f"Not enough unique intersection points ({len(cross_section_vertices)}) "
                  f"to form a {target_hull_dimension}D convex hull.")
            # For 3D printing, a 1D or 2D result from a 3D slice might not be a "mesh"
            # For example, cross-section of a 3D cube at its corner might just be a point
            # or a line if it hits an edge. We return the vertices, but no faces.
            return cross_section_vertices, None

        try:
            # Compute the convex hull of the projected intersection points
            # hull.points will be the vertices passed to ConvexHull (cross_section_vertices)
            # hull.simplices will be the indices into hull.points (cross_section_vertices)
            hull = ConvexHull(cross_section_vertices)

            # The `simplices` attribute gives the indices of the vertices that form each face (triangle)
            return cross_section_vertices, hull.simplices
        except Exception as e:
            print(f"Error computing convex hull: {e}")
            print(f"Intersection points for hull computation: {cross_section_vertices}")
            return cross_section_vertices, None  # Return points even if hull fails

    def get_k_faces(self, k):
        """
        Returns the set of k-dimensional Face objects.
        Args:
            k (int): The dimension of faces to retrieve.
        Returns:
            set: A set of Face objects, or an empty set if no faces of that dimension exist.
        """
        return self.k_faces.get(k, set())

    def get_vertex_coordinates(self):
        """
        Returns a dictionary mapping initial vertex indices to their current numpy array coordinates.
        """
        return {idx: vertex.coordinates for idx, vertex in self.vertices_map.items()}

    def get_edges_as_pairs(self):
        """
        Returns a list of (vertex_idx1, vertex_idx2) tuples for all 1-dimensional edges.
        Each tuple contains the initial_indices of the two vertices forming the edge, sorted.
        """
        edge_pairs = []
        if 1 in self.k_faces:
            for edge_face in self.k_faces[1]:
                edge_pairs.append(tuple(sorted(list(edge_face.vertices_indices))))
        return edge_pairs

    def get_faces_as_vertex_indices(self):
        """
        Returns a list of frozensets of initial vertex indices for all 2-dimensional faces (squares).
        """
        face_sets = []
        if 2 in self.k_faces:
            for face_obj in self.k_faces[2]:
                face_sets.append(face_obj.vertices_indices)
        return face_sets
