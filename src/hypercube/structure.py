import numpy as np
import itertools
from itertools import combinations


# --- Vertex Class ---
class Vertex:
    def __init__(self, coordinates, initial_index=None):
        self.coordinates = np.array(coordinates, dtype=float)
        self.initial_index = initial_index  # To track original position in hypercube
        # Add other potential attributes here:
        self.color = None
        self.highlighted = False

    def __repr__(self):
        return f"Vertex(id={self.initial_index}, coords={self.coordinates.tolist()})"

    def __hash__(self):
        # Hash based on initial_index if available, otherwise coordinates (less robust if coords change)
        return hash(self.initial_index) if self.initial_index is not None else hash(tuple(self.coordinates.tolist()))

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return NotImplemented
        if self.initial_index is not None and other.initial_index is not None:
            return self.initial_index == other.initial_index
        return np.array_equal(self.coordinates, other.coordinates)


# --- Face Class ---
class Face:
    def __init__(self, vertices_indices, dimension, all_vertices_map):
        self.vertices_indices = frozenset(vertices_indices)
        self.dimension = dimension
        self._all_vertices_map = all_vertices_map

    @property
    def vertices(self):
        return [self._all_vertices_map[idx] for idx in self.vertices_indices]

    def __repr__(self):
        return f"Face(dim={self.dimension}, indices={sorted(list(self.vertices_indices))})"

    def __hash__(self):
        return hash((self.dimension, self.vertices_indices))

    def __eq__(self, other):
        return isinstance(other, Face) and \
            self.dimension == other.dimension and \
            self.vertices_indices == other.vertices_indices


# --- Hypercube Class ---
class Hypercube:
    def __init__(self, dimension):
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError("Dimension must be a non-negative integer.")
        self.dimension = dimension
        self.vertices_map = {}  # Maps initial_index to Vertex object
        self.k_faces = {}  # Dictionary to store sets of k-dimensional faces

        self._generate_vertices()
        self._generate_k_faces()

    def _generate_vertices(self):
        """Generates all vertices of the n-dimensional hypercube."""
        interval = [0, 1]
        coordinates = np.array(
            [[val for val in prod] for prod in itertools.product(interval, repeat=self.dimension)]
        )
        for i, coord in enumerate(coordinates):
            self.vertices_map[i] = Vertex(coord, initial_index=i)
        self.k_faces[0] = {Face({i}, 0, self.vertices_map) for i in self.vertices_map}

    def _generate_k_faces(self):
        """Generates k-dimensional faces using the combinatorial definition."""
        if self.dimension == 0:
            return

        for d in range(1, self.dimension + 1):
            current_dim_faces = set()
            for varying_dims_indices in combinations(range(self.dimension), d):
                fixed_dims_indices = [i for i in range(self.dimension) if i not in varying_dims_indices]

                for fixed_values_tuple in itertools.product([0, 1], repeat=len(fixed_dims_indices)):
                    face_vertex_indices = set()
                    for varying_values_tuple in itertools.product([0, 1], repeat=len(varying_dims_indices)):
                        vertex_coords = [None] * self.dimension
                        v_idx_counter = 0
                        f_idx_counter = 0
                        for dim_idx in range(self.dimension):
                            if dim_idx in varying_dims_indices:
                                vertex_coords[dim_idx] = varying_values_tuple[v_idx_counter]
                                v_idx_counter += 1
                            else:
                                vertex_coords[dim_idx] = fixed_values_tuple[f_idx_counter]
                                f_idx_counter += 1

                        # Find the initial_index of this vertex
                        for idx, v_obj in self.vertices_map.items():
                            if np.array_equal(v_obj.coordinates, np.array(vertex_coords)):
                                face_vertex_indices.add(idx)
                                break
                    current_dim_faces.add(Face(face_vertex_indices, d, self.vertices_map))
            self.k_faces[d] = current_dim_faces

    def translate(self, vector):
        """Translates the hypercube by a given vector."""
        if len(vector) != self.dimension:
            raise ValueError("Translation vector must match hypercube dimension.")
        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates += np.array(vector, dtype=float)

    def rotate(self, plane_indices, angle):
        """
        Rotates the hypercube in a specified 2D plane (e.g., [0, 1] for XY plane).
        The rotation matrix is applied to the coordinates in the chosen plane.
        """
        if len(plane_indices) != 2 or not all(0 <= i < self.dimension for i in plane_indices):
            raise ValueError(f"Plane indices must be a pair of integers within [0, {self.dimension - 1}].")

        i, j = plane_indices
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        rotation_matrix = np.identity(self.dimension)
        rotation_matrix[i, i] = cos_theta
        rotation_matrix[i, j] = -sin_theta
        rotation_matrix[j, i] = sin_theta
        rotation_matrix[j, j] = cos_theta

        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates = np.dot(rotation_matrix, vertex_obj.coordinates)

    def project(self, target_dimension, projection_type='orthogonal', perspective_params=None):
        """
        Projects the hypercube onto a lower-dimensional space.

        Args:
            target_dimension (int): The dimension of the space to project onto.
            projection_type (str): 'orthogonal' or 'perspective'.
            perspective_params (dict): Dictionary of parameters for perspective projection.
                                       For 'perspective', this could include 'view_distance'.

        Returns:
            dict: A dictionary mapping initial vertex indices to new Vertex objects
                  with projected coordinates.
        """
        if target_dimension >= self.dimension or target_dimension < 0:
            raise ValueError("Target dimension must be less than current dimension and non-negative.")

        projected_vertices = {}

        if projection_type == 'orthogonal':
            # Orthogonal projection: simply drop the higher dimensions
            for idx, vertex_obj in self.vertices_map.items():
                projected_coords = vertex_obj.coordinates[:target_dimension]
                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        elif projection_type == 'perspective':
            if perspective_params is None or 'view_distance' not in perspective_params:
                raise ValueError("Perspective projection requires 'view_distance' in perspective_params.")

            view_distance = float(perspective_params['view_distance'])
            if view_distance <= 0:
                raise ValueError("View distance for perspective projection must be positive.")

            # Simple perspective projection (e.g., for projecting N-D to N-1D by W-axis)
            # This projects all points (x_0, x_1, ..., x_N-1) onto the x_N-1 = 0 plane.
            # The last coordinate is treated as the 'depth' axis.
            # (x, y, z, w) -> (x_p, y_p, z_p) where x_p = x / (1 - w/d), etc.
            # Here, we'll assume the 'view_distance' is applied to the last `self.dimension - target_dimension` axes.

            for idx, vertex_obj in self.vertices_map.items():
                original_coords = vertex_obj.coordinates
                projected_coords = np.zeros(target_dimension)

                # For simplicity, project from self.dimension to target_dimension
                # by treating remaining dimensions as 'depth' axes.
                # A common perspective projection from N to N-1 (e.g., 4D to 3D):
                # (x,y,z,w) -> (x/(1-w/d), y/(1-w/d), z/(1-w/d))
                # Generalizing: perspective division by (1 - sum_of_higher_dims / view_distance)

                # A more robust general N-D to M-D perspective projection is complex.
                # Let's consider the standard projection from N to N-1 by dropping the N-th coordinate
                # with perspective along that dropped axis.

                # If target_dimension is self.dimension - 1:
                if target_dimension == self.dimension - 1:
                    last_coord = original_coords[self.dimension - 1]
                    divisor = (view_distance - last_coord)

                    if divisor == 0:
                        # Handle division by zero - points on the projection plane
                        # Could clip, move to infinity, or assign a large value
                        # For visualization, clipping or setting to large magnitude might be appropriate.
                        # For now, let's just make it a very large number for demonstrative purposes
                        divisor = 1e-6  # Small non-zero value to avoid inf/nan

                    for i in range(target_dimension):
                        projected_coords[i] = original_coords[i] * view_distance / divisor
                else:
                    # For projecting from N-D to M-D where M < N-1,
                    # a specific projection matrix or iterative application might be needed.
                    # For simplicity, if target_dimension < self.dimension - 1,
                    # we'll still use the last dimension as the perspective axis,
                    # and then just truncate remaining dimensions, making it a hybrid.
                    # A true N-D to M-D perspective is more involved.

                    last_coord = original_coords[self.dimension - 1]
                    divisor = (view_distance - last_coord)

                    if divisor == 0:
                        divisor = 1e-6

                    for i in range(target_dimension):
                        projected_coords[i] = original_coords[i] * view_distance / divisor

                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}. Choose 'orthogonal' or 'perspective'.")

        return projected_vertices

    def unfold(self, unfolding_type='default', step=0):
        """
        Conceptual method for unfolding the hypercube.
        The output format would depend heavily on the visualization target.

        Args:
            unfolding_type (str): Specifies which unfolding to perform (e.g., 'cross_tesseract', 'net').
                                  This parameter would control the "unfolding strategy".
            step (float): A parameter for animating the unfolding process, from 0 (closed) to 1 (fully unfolded).
                          This would apply a transformation based on the unfolding chosen.

        Returns:
            A representation of the unfolded hypercube's components,
            e.g., a list of Face objects with transformed coordinates, or a flat 2D net.
            For AR/VR, this might be a list of 3D models or transforms.
        """
        # This method is highly conceptual and would require significant
        # mathematical definition of specific unfoldings.

        # Example: Unfolding a 3D cube into a 2D net (a cross shape)
        if self.dimension == 3 and unfolding_type == 'net':
            if not (0 <= step <= 1):
                raise ValueError("Step for unfolding must be between 0 and 1.")

            unfolded_vertices = {idx: Vertex(v.coordinates.copy(), v.initial_index)
                                 for idx, v in self.vertices_map.items()}

            # Define faces by their fixed axis and value (0 or 1)
            # For a unit cube [0,1]^3
            # x=0 face: indices {0, 1, 2, 3}
            # x=1 face: indices {4, 5, 6, 7}
            # y=0 face: indices {0, 1, 4, 5}
            # y=1 face: indices {2, 3, 6, 7}
            # z=0 face: indices {0, 2, 4, 6}
            # z=1 face: indices {1, 3, 5, 7}

            # This is a simplified example for a 3D cube's standard net unfolding:
            # Imagine the x=1 face (4,5,6,7) as the central face.
            # Then unfold others around it.

            # The actual transformations would be rotations of faces around shared edges.
            # For a step `s`, rotate by `s * 90` degrees.

            # Example: Unfold the y=0 face (0,1,4,5) off the x=1 face (4,5,6,7) around edge (4,5)
            # The rotation axis is the edge connecting common vertices (e.g., (4,5)).
            # The rotation angle depends on 'step'.

            # For a zoetrope, you might unfold a tesseract into 3D cells, then animate those.
            # Each cell would be a 3D cube. The 'unfolding_type' could specify
            # which 3D cube is the 'center' and how the others attach.

            print(f"Unfolding type '{unfolding_type}' at step {step} for a {self.dimension}-D hypercube is conceptual.")
            print("This would involve defining specific rotation axes and angles for each k-face.")

            # Placeholder for returning transformed face coordinates
            transformed_faces = {}
            for k, faces_set in self.k_faces.items():
                transformed_faces[k] = []
                for face_obj in faces_set:
                    # In a real implementation, apply transformations based on unfolding_type and step
                    # This would involve rotations around shared edges, displacing the face vertices.
                    transformed_coords = {idx: unfolded_vertices[idx].coordinates for idx in face_obj.vertices_indices}
                    transformed_faces[k].append({
                        'indices': list(face_obj.vertices_indices),
                        'coords': [transformed_coords[idx].tolist() for idx in face_obj.vertices_indices]
                    })
            return transformed_faces

        elif self.dimension == 4 and unfolding_type == 'cross_tesseract':
            # This is the famous 'tesseract net' projection from 4D to 3D
            # It involves selecting one 'cell' (3D cube) as the central one,
            # and then "unfolding" the other 7 cells around it.
            # Each cell would effectively be translated and rotated in 3D space.

            print(f"Conceptual unfolding for 4D hypercube (tesseract net) at step {step}.")
            print("This requires specific mapping of 3D cells and their rotations/translations.")

            # This would involve selecting one 'central' 3-face (cube)
            # and then rotating the other 3-faces around their common 2-faces (squares)
            # into the 3D space of the zoetrope.

            # The result would be a collection of 3D cube models (8 of them)
            # positioned and rotated in a way that represents the unfolded state.

            # Placeholder for returning transformed cell coordinates
            transformed_cells = []
            if 3 in self.k_faces:
                for cell_face in self.k_faces[3]:
                    # Apply complex transformations here based on unfolding_type and step
                    cell_transformed_vertices = {}
                    for v_idx in cell_face.vertices_indices:
                        # For a real unfolding, these coordinates would be calculated
                        # based on the unfolding process (e.g., rotating entire cubes)
                        cell_transformed_vertices[v_idx] = self.vertices_map[v_idx].coordinates.copy()
                    transformed_cells.append({
                        'indices': list(cell_face.vertices_indices),
                        'coords': [cell_transformed_vertices[idx].tolist() for idx in cell_face.vertices_indices]
                    })
            return transformed_cells


        else:
            print(f"Unfolding type '{unfolding_type}' for {self.dimension}-D hypercube not implemented or recognized.")
            return None

    def get_k_faces(self, k):
        """Returns the set of k-dimensional faces (Face objects)."""
        return self.k_faces.get(k, set())

    def get_vertex_coordinates(self):
        """Returns a dictionary of initial vertex indices to their current coordinates."""
        return {idx: vertex.coordinates for idx, vertex in self.vertices_map.items()}

    def get_edges_as_pairs(self):
        """Returns a list of (vertex_idx1, vertex_idx2) tuples for edges."""
        edge_pairs = []
        # Ensure k_faces[1] exists for edges
        if 1 in self.k_faces:
            for edge_face in self.k_faces[1]:
                # It's a frozenset of 2 indices, convert to sorted tuple for consistent representation
                edge_pairs.append(tuple(sorted(list(edge_face.vertices_indices))))
        return edge_pairs

    def get_faces_as_vertex_indices(self):
        """Returns a list of frozensets of vertex indices for 2D faces."""
        face_sets = []
        if 2 in self.k_faces:
            for face_obj in self.k_faces[2]:
                face_sets.append(face_obj.vertices_indices)
        return face_sets

# You can now instantiate and use:
# cube = Hypercube(3)
# tesseract = Hypercube(4)