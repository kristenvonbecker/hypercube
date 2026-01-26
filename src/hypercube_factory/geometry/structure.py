"""Geometry primitives and transformations."""

import numpy as np
import itertools
from dataclasses import dataclass, field
from itertools import combinations
from scipy.spatial import ConvexHull


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
        # Fallback if initial_index is not set for some reason.
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
        self.vertices_map = {}  # Maps initial_index to Vertex object.
        self.k_faces = {}  # Dictionary to store sets of k-dimensional faces.

        self._generate_vertices()
        self._generate_k_faces()

    def _generate_vertices(self):
        """Generates all 2^N vertices of the N-dimensional hypercube."""
        interval = [0, 1]  # Vertices are at (0,0,..,0) to (1,1,..,1).
        coordinates = np.array(
            [[val for val in prod] for prod in itertools.product(interval, repeat=self.dimension)]
        )
        for i, coord in enumerate(coordinates):
            self.vertices_map[i] = Vertex(coord, initial_index=i)
        # 0-dimensional faces are just the vertices themselves.
        self.k_faces[0] = {Face({i}, 0, self.vertices_map) for i in self.vertices_map}

    def _generate_k_faces(self):
        """
        Generates k-dimensional faces for all k from 1 to self.dimension.
        A k-face is defined by holding (N-k) coordinates fixed and letting k coordinates vary.
        """
        if self.dimension == 0:
            return  # A 0-dimensional hypercube has only 0-faces (vertices).

        for d in range(1, self.dimension + 1):
            current_dim_faces = set()
            # Iterate through all combinations of 'd' dimensions that will vary.
            for varying_dims_indices in combinations(range(self.dimension), d):
                fixed_dims_indices = [i for i in range(self.dimension) if i not in varying_dims_indices]

                # Iterate through all 2^(N-d) combinations of fixed values for the fixed dimensions.
                for fixed_values_tuple in itertools.product([0, 1], repeat=len(fixed_dims_indices)):
                    face_vertex_indices = set()
                    # For each combination of fixed dimensions/values, generate the 2^d vertices of this face.
                    for varying_values_tuple in itertools.product([0, 1], repeat=len(varying_dims_indices)):
                        vertex_coords = [None] * self.dimension
                        v_idx_counter = 0  # Counter for varying_values_tuple.
                        f_idx_counter = 0  # Counter for fixed_values_tuple.

                        # Reconstruct the full N-dimensional coordinates for this vertex.
                        for dim_idx in range(self.dimension):
                            if dim_idx in varying_dims_indices:
                                vertex_coords[dim_idx] = varying_values_tuple[v_idx_counter]
                                v_idx_counter += 1
                            else:
                                vertex_coords[dim_idx] = fixed_values_tuple[f_idx_counter]
                                f_idx_counter += 1

                        # Find the initial_index of this vertex in the global map.
                        # This lookup ensures we use the canonical Vertex object for the face.
                        for idx, v_obj in self.vertices_map.items():
                            if np.array_equal(v_obj.coordinates, np.array(vertex_coords)):
                                face_vertex_indices.add(idx)
                                break
                    current_dim_faces.add(Face(face_vertex_indices, d, self.vertices_map))
            self.k_faces[d] = current_dim_faces

    def translate(self, vector):
        """
        Translates the hypercube and returns the instance for method chaining.
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Translation vector must match hypercube dimension ({self.dimension}).")
        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates += np.array(vector, dtype=float)
        return self

    def rotate(self, plane_indices, angle):
        """
        Rotates the hypercube and returns the instance for method chaining.
        """
        if len(plane_indices) != 2 or not all(0 <= i < self.dimension for i in plane_indices):
            raise ValueError(f"Plane indices must be a pair of integers within [0, {self.dimension - 1}].")

        i, j = plane_indices
        if i == j:
            raise ValueError("Plane indices must refer to two distinct dimensions.")
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        rotation_matrix = np.identity(self.dimension)
        rotation_matrix[i, i] = cos_theta
        rotation_matrix[i, j] = -sin_theta
        rotation_matrix[j, i] = sin_theta
        rotation_matrix[j, j] = cos_theta

        for vertex_obj in self.vertices_map.values():
            vertex_obj.coordinates = np.dot(rotation_matrix, vertex_obj.coordinates)
        return self

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
            # Orthogonal projection: simply truncate coordinates to the target_dimension.
            for idx, vertex_obj in self.vertices_map.items():
                projected_coords = vertex_obj.coordinates[:target_dimension]
                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        elif projection_type == 'perspective':
            if perspective_params is None or 'view_distance' not in perspective_params:
                raise ValueError("Perspective projection requires 'view_distance' in perspective_params.")

            view_distance = float(perspective_params['view_distance'])
            if view_distance <= 0:
                raise ValueError("View distance for perspective projection must be positive.")

            # Simple perspective projection (e.g., projecting from N-D to N-1D by treating the last dimension as a 'depth' axis).
            # (x_0, ..., x_{N-2}, x_{N-1}) -> (x_0_p, ..., x_{N-2}_p)
            # where x_i_p = x_i * view_distance / (view_distance - x_{N-1}).

            for idx, vertex_obj in self.vertices_map.items():
                original_coords = vertex_obj.coordinates.copy()
                projected_coords = np.zeros(target_dimension)

                # Determine the 'depth' coordinate for perspective effect.
                # Use the last dimension of the original hypercube for perspective division.
                if self.dimension > 0:  # Ensure there's a last coordinate to use.
                    depth_coord = original_coords[self.dimension - 1]
                else:  # For 0D hypercube (a point), no perspective applies.
                    depth_coord = 0.0

                divisor = (view_distance - depth_coord)

                if divisor == 0:
                    # Handle division by zero (points on the projection plane).
                    # (Clip these or place them far away.)
                    divisor = 1e-9  # A very small epsilon to avoid division by zero.

                # Apply perspective effect to the first `target_dimension` coordinates.
                for i in range(target_dimension):
                    # Only apply perspective if original dimension has more than `i` coordinates.
                    if i < self.dimension:
                        projected_coords[i] = original_coords[i] * view_distance / divisor
                    else:
                        projected_coords[i] = 0.0

                projected_vertices[idx] = Vertex(projected_coords, initial_index=idx)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}. Choose 'orthogonal' or 'perspective'.")

        return projected_vertices

    def cross_section(self, hyperplane_dim=None, hyperplane_value=0.0):
        """
        Calculates the cross-section of the hypercube with a given hyperplane.
        Assumes the hyperplane is defined by holding one dimension fixed at a certain value.
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
        original_edges = self.get_edges_as_pairs()  # Get edges based on initial_indices.

        # Map current vertex initial_indices to their current coordinates.
        current_vertex_coords = self.get_vertex_coordinates()

        for idx1, idx2 in original_edges:
            v1_coords = current_vertex_coords[idx1]
            v2_coords = current_vertex_coords[idx2]

            coord1_val = v1_coords[hyperplane_dim]
            coord2_val = v2_coords[hyperplane_dim]

            # Check if the edge crosses the hyperplane.
            # Case 1: One endpoint is on the hyperplane.
            # Use np.isclose for floating point comparisons.
            is_v1_on_hyperplane = np.isclose(coord1_val, hyperplane_value)
            is_v2_on_hyperplane = np.isclose(coord2_val, hyperplane_value)

            if is_v1_on_hyperplane and is_v2_on_hyperplane:
                # Both endpoints are on the hyperplane, the entire edge is part of the cross-section.
                # We add both endpoints. Duplicate handling will occur later.
                intersection_points_full_dim.append(v1_coords)
                intersection_points_full_dim.append(v2_coords)
            elif is_v1_on_hyperplane:
                # Only v1 is on the hyperplane.
                intersection_points_full_dim.append(v1_coords)
            elif is_v2_on_hyperplane:
                # Only v2 is on the hyperplane.
                intersection_points_full_dim.append(v2_coords)
            # Case 2: The hyperplane intersects the interior of the edge.
            elif (coord1_val - hyperplane_value) * (coord2_val - hyperplane_value) < 0:
                delta = coord2_val - coord1_val
                if np.isclose(delta, 0):  # Should not happen if signs are different, but for safety.
                    continue
                t = (hyperplane_value - coord1_val) / delta
                intersection_coord_full_dim = v1_coords + t * (v2_coords - v1_coords)
                intersection_points_full_dim.append(intersection_coord_full_dim)

        if not intersection_points_full_dim:
            print(f"No intersection points found for hyperplane dim {hyperplane_dim} at value {hyperplane_value}.")
            return None, None

        # Convert to numpy array and remove duplicates using rounding for float precision.
        # Convert each point to a tuple for hashing in a set, round to avoid floating point issues.
        unique_intersection_points_full_dim = np.array(list(
            {tuple(p.round(8)) for p in intersection_points_full_dim}
        ))

        # Project the unique intersection points onto the (N-1)D space by deleting the hyperplane_dim.
        cross_section_vertices = np.delete(unique_intersection_points_full_dim, hyperplane_dim, axis=1)

        target_hull_dimension = self.dimension - 1

        if len(cross_section_vertices) < target_hull_dimension + 1:
            print(f"Not enough unique intersection points ({len(cross_section_vertices)}) "
                  f"to form a {target_hull_dimension}D convex hull.")
            # For 3D printing, a 1D or 2D result from a 3D slice might not be a "mesh."
            # For example, a cross-section of a 3D cube at its corner might be a point or a line.
            # We return the vertices, but no faces.
            return cross_section_vertices, None

        try:
            # Compute the convex hull of the projected intersection points.
            # hull.points will be the vertices passed to ConvexHull (cross_section_vertices).
            # hull.simplices will be the indices into hull.points (cross_section_vertices).
            hull = ConvexHull(cross_section_vertices)

            # The `simplices` attribute gives the indices of the vertices that form each face (triangle).
            return cross_section_vertices, hull.simplices
        except Exception as e:
            print(f"Error computing convex hull: {e}")
            print(f"Intersection points for hull computation: {cross_section_vertices}")
            return cross_section_vertices, None  # Return points even if hull fails.

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


# --- Hypercube Net Structures ---
@dataclass
class Hinge:
    """
    Describes a hinge between two k-faces along their shared (k-1)-face.
    axis_basis: Orthonormal basis spanning the hinge subspace in ambient N-D.
    plane_basis: Two orthonormal vectors spanning the fold plane (orthogonal to axis_basis).
    """

    shared_vertices_indices: frozenset
    axis_basis: np.ndarray
    plane_basis: np.ndarray
    pivot: np.ndarray


@dataclass
class NetFace:
    """
    Represents a k-face in a hypercube net.
    vertex_positions are local copies of the hypercube's current vertex coordinates.
    """

    face_id: int
    dimension: int
    vertices_indices: frozenset
    vertex_positions: dict
    parent_id: int | None = None
    children_ids: list = field(default_factory=list)
    hinge_to_parent: Hinge | None = None
    fold_angle: float = 0.0


class HypercubeNet:
    """
    Represents a hypercube net as a tree of k-faces connected by hinges.
    Provides fold/unfold operations by rotating a face subtree around its hinge.
    """

    def __init__(self, hypercube, face_dimension, root_face_id=0):
        if not isinstance(hypercube, Hypercube):
            raise TypeError("hypercube must be an instance of Hypercube.")
        if not (0 <= face_dimension <= hypercube.dimension):
            raise ValueError("face_dimension must be within [0, hypercube.dimension].")

        self.hypercube_dim = hypercube.dimension
        self.face_dim = face_dimension
        self.faces = {}
        self.root_face_id = root_face_id

        self._build_from_hypercube(hypercube)

    def _build_from_hypercube(self, hypercube):
        k_faces = list(hypercube.get_k_faces(self.face_dim))
        if not k_faces:
            raise ValueError(f"No faces of dimension {self.face_dim} exist for this hypercube.")

        # Map each k-face to an id using its vertex indices as a stable key.
        face_key_to_id = {}
        for face_id, face in enumerate(k_faces):
            face_key_to_id[frozenset(face.vertices_indices)] = face_id

        # Snapshot vertex positions from the hypercube at build time.
        current_coords = hypercube.get_vertex_coordinates()

        for face_id, face in enumerate(k_faces):
            vertex_positions = {
                idx: current_coords[idx].copy()
                for idx in face.vertices_indices
            }
            self.faces[face_id] = NetFace(
                face_id=face_id,
                dimension=self.face_dim,
                vertices_indices=frozenset(face.vertices_indices),
                vertex_positions=vertex_positions,
            )

        # Build adjacency based on shared (k-1)-faces.
        adjacency = {face_id: [] for face_id in self.faces}
        if self.face_dim == 0:
            self.root_face_id = 0
            return

        # Two k-faces are adjacent if they share exactly 2^(k-1) vertices.
        required_shared_vertices = 2 ** (self.face_dim - 1)
        face_ids = list(self.faces.keys())
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                f_id = face_ids[i]
                g_id = face_ids[j]
                shared = self.faces[f_id].vertices_indices & self.faces[g_id].vertices_indices
                if len(shared) == required_shared_vertices:
                    adjacency[f_id].append(g_id)
                    adjacency[g_id].append(f_id)

        # Build a spanning tree (BFS) rooted at root_face_id.
        if self.root_face_id not in self.faces:
            self.root_face_id = 0

        visited = set()
        queue = [self.root_face_id]
        visited.add(self.root_face_id)

        while queue:
            parent_id = queue.pop(0)
            for child_id in adjacency[parent_id]:
                if child_id in visited:
                    continue
                visited.add(child_id)
                queue.append(child_id)
                self.faces[child_id].parent_id = parent_id
                self.faces[parent_id].children_ids.append(child_id)

                # Build the hinge from the shared (k-1)-face.
                hinge = self._build_hinge(self.faces[parent_id], self.faces[child_id])
                self.faces[child_id].hinge_to_parent = hinge

    def _build_hinge(self, parent_face, child_face):
        shared = parent_face.vertices_indices & child_face.vertices_indices
        if not shared:
            raise ValueError("Faces are not adjacent; cannot build hinge.")

        # Hinge axis basis comes from the shared (k-1)-face.
        shared_coords = np.array(
            [parent_face.vertex_positions[idx] for idx in shared]
        )
        axis_basis = self._compute_basis(shared_coords, self.face_dim - 1)

        # Compute full k-face bases to find the two fold-plane directions.
        parent_coords = np.array(
            [parent_face.vertex_positions[idx] for idx in parent_face.vertices_indices]
        )
        child_coords = np.array(
            [child_face.vertex_positions[idx] for idx in child_face.vertices_indices]
        )
        parent_basis = self._compute_basis(parent_coords, self.face_dim)
        child_basis = self._compute_basis(child_coords, self.face_dim)

        # Pick one vector orthogonal to the hinge from each face.
        u = self._vector_orthogonal_to_basis(parent_basis, axis_basis)
        v = self._vector_orthogonal_to_basis(child_basis, axis_basis)
        # Orthogonalize to get a stable fold plane basis.
        w = self._make_orthonormal_to(u, v, axis_basis)

        # Any point on the hinge works as the pivot; centroid is stable.
        pivot = shared_coords.mean(axis=0)
        plane_basis = np.vstack([u, w])

        return Hinge(
            shared_vertices_indices=frozenset(shared),
            axis_basis=axis_basis,
            plane_basis=plane_basis,
            pivot=pivot,
        )

    def _compute_basis(self, coords, target_dim):
        """
        Computes an orthonormal basis for the subspace spanned by coords.
        Uses SVD for stability. Returns shape (target_dim, N).
        """
        if target_dim == 0:
            return np.zeros((0, self.hypercube_dim))
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:target_dim]
        return basis

    def _vector_orthogonal_to_basis(self, candidate_basis, axis_basis):
        """
        Finds a unit vector in candidate_basis that is orthogonal to axis_basis.
        """
        for vec in candidate_basis:
            v = vec.copy()
            for axis_vec in axis_basis:
                v -= np.dot(v, axis_vec) * axis_vec
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                return v / norm
        raise ValueError("Could not find a vector orthogonal to the hinge axis.")

    def _make_orthonormal_to(self, u, v, axis_basis):
        """
        Makes a unit vector orthogonal to u and axis_basis, using v as a guide.
        """
        w = v - np.dot(v, u) * u
        for axis_vec in axis_basis:
            w -= np.dot(w, axis_vec) * axis_vec
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            return w / norm

        # Fallback: pick any vector orthogonal to axis_basis and u.
        ortho_basis = np.vstack([axis_basis, u])
        _, _, vt = np.linalg.svd(ortho_basis, full_matrices=True)
        for candidate in vt:
            if np.allclose(ortho_basis @ candidate, 0, atol=1e-8):
                return candidate / np.linalg.norm(candidate)
        raise ValueError("Could not construct an orthonormal fold-plane vector.")

    def _rotation_in_plane(self, e1, e2, angle):
        """
        Builds an N-D rotation matrix that rotates in the plane spanned by e1 and e2.
        """
        n = self.hypercube_dim
        r = np.identity(n)
        e1 = e1.reshape(n, 1)
        e2 = e2.reshape(n, 1)
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)

        r += sin_theta * (e2 @ e1.T - e1 @ e2.T)
        r += (cos_theta - 1.0) * (e1 @ e1.T + e2 @ e2.T)
        return r

    def rotate_subtree(self, face_id, angle):
        """
        Rotates a face and its descendants by angle (radians) around its hinge.
        """
        face = self.faces.get(face_id)
        if face is None:
            raise ValueError(f"Unknown face_id: {face_id}")
        if face.parent_id is None or face.hinge_to_parent is None:
            raise ValueError("Root face cannot be rotated (no hinge to parent).")

        hinge = face.hinge_to_parent
        e1, e2 = hinge.plane_basis
        rotation = self._rotation_in_plane(e1, e2, angle)
        pivot = hinge.pivot

        for sub_id in self._iter_subtree(face_id):
            self._apply_rotation_to_face(self.faces[sub_id], rotation, pivot)

        face.fold_angle += angle

    def _apply_rotation_to_face(self, face, rotation, pivot):
        # Rotate each stored vertex coordinate about the hinge pivot.
        for idx, coord in face.vertex_positions.items():
            face.vertex_positions[idx] = pivot + rotation @ (coord - pivot)

    def _iter_subtree(self, face_id):
        stack = [face_id]
        while stack:
            current_id = stack.pop()
            yield current_id
            stack.extend(self.faces[current_id].children_ids)

    def get_face_vertices(self, face_id):
        face = self.faces.get(face_id)
        if face is None:
            raise ValueError(f"Unknown face_id: {face_id}")
        return np.array([face.vertex_positions[idx] for idx in face.vertices_indices])


class TesseractNet3D:
    """
    Builds 3D tesseract nets as polycubes (all vertices lie in 3D; w=0 implicitly).
    Net styles define distinct layouts of the 8 cube cells.
    """

    def __init__(self, net_style="standard", cube_size=1.0):
        if cube_size <= 0:
            raise ValueError("cube_size must be positive.")
        self.net_style = net_style
        self.cube_size = float(cube_size)
        self.faces = {}
        self.root_face_id = 0
        self._build()

    def _cube_vertices(self, origin):
        x0, y0, z0 = origin
        s = self.cube_size
        return np.array([
            [x0, y0, z0],
            [x0 + s, y0, z0],
            [x0, y0 + s, z0],
            [x0 + s, y0 + s, z0],
            [x0, y0, z0 + s],
            [x0 + s, y0, z0 + s],
            [x0, y0 + s, z0 + s],
            [x0 + s, y0 + s, z0 + s],
        ], dtype=float)

    def _net_origins(self):
        s = self.cube_size
        if self.net_style == "standard":
            # Central cube with 6 neighbors, plus one extra on +Z.
            return [
                (0, 0, 0),          # center
                (s, 0, 0),          # +X
                (-s, 0, 0),         # -X
                (0, s, 0),          # +Y
                (0, -s, 0),         # -Y
                (0, 0, s),          # +Z
                (0, 0, -s),         # -Z
                (0, 0, 2 * s),      # extra on +Z
            ]
        if self.net_style == "zigzag":
            # A more believable layout: 5-cube chain with two side cubes.
            # This stays tree-like and looks closer to common tesseract nets.
            return [
                (0, 0, 0),          # root
                (s, 0, 0),          # chain 1
                (2 * s, 0, 0),      # chain 2
                (3 * s, 0, 0),      # chain 3
                (4 * s, 0, 0),      # chain 4
                (2 * s, s, 0),      # side off chain
                (2 * s, -s, 0),     # side off chain
                (1 * s, 0, s),      # out-of-plane leaf
            ]
        raise ValueError(f"Unknown net_style: {self.net_style}. Use 'standard' or 'zigzag'.")

    def _build(self):
        origins = self._net_origins()
        for face_id, origin in enumerate(origins):
            vertices = self._cube_vertices(origin)
            vertex_positions = {idx: vertices[idx] for idx in range(vertices.shape[0])}
            self.faces[face_id] = NetFace(
                face_id=face_id,
                dimension=3,
                vertices_indices=frozenset(vertex_positions.keys()),
                vertex_positions=vertex_positions,
            )

        # Build adjacency based on shared faces (origins differ by one cube along one axis).
        adjacency = {face_id: [] for face_id in self.faces}
        s = self.cube_size
        origins_arr = np.array(origins, dtype=float)
        for i in range(len(origins)):
            for j in range(i + 1, len(origins)):
                delta = origins_arr[j] - origins_arr[i]
                matches = np.isclose(np.abs(delta), s) | np.isclose(delta, 0.0)
                if np.all(matches) and np.isclose(np.sum(np.isclose(np.abs(delta), s)), 1):
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        # Build a spanning tree for future folding logic.
        visited = set()
        queue = [self.root_face_id]
        visited.add(self.root_face_id)
        while queue:
            parent_id = queue.pop(0)
            for child_id in adjacency[parent_id]:
                if child_id in visited:
                    continue
                visited.add(child_id)
                queue.append(child_id)
                self.faces[child_id].parent_id = parent_id
                self.faces[parent_id].children_ids.append(child_id)


# --- Hypercube Net Enumeration (Facet Nets) ---
@dataclass
class UnfoldedNet:
    """
    Stores an unfolded net as an (n-1)-dimensional object.
    face_vertices: dict[face_id] -> np.ndarray of shape (2^(n-1), n-1)
    """

    dimension: int
    face_vertices: dict
    tree_edges: list
    root_face_id: int = 0


class _DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True

    def copy(self):
        clone = _DisjointSet(0)
        clone.parent = self.parent[:]
        clone.rank = self.rank[:]
        clone.count = self.count
        return clone


class HypercubeNetEnumerator:
    """
    Enumerates spanning trees of facet adjacency and filters those that unfold to a valid net.
    Unfolded nets are represented in (n-1)-dimensional coordinates (root-face plane).
    """

    def __init__(self, hypercube, face_dimension=None):
        if not isinstance(hypercube, Hypercube):
            raise TypeError("hypercube must be an instance of Hypercube.")
        self.hypercube = hypercube
        self.n = hypercube.dimension
        self.face_dim = self.n - 1 if face_dimension is None else face_dimension
        if self.face_dim != self.n - 1:
            raise ValueError("Only facet nets (face_dimension = n-1) are supported.")
        if self.n < 2:
            raise ValueError("Facet nets require hypercube dimension >= 2.")

        self.faces = list(hypercube.get_k_faces(self.face_dim))
        if not self.faces:
            raise ValueError(f"No faces of dimension {self.face_dim} exist for this hypercube.")

        self.face_id_by_key = {
            frozenset(face.vertices_indices): idx
            for idx, face in enumerate(self.faces)
        }
        self.face_vertices_indices = [face.vertices_indices for face in self.faces]
        self.face_positions_nd = [
            np.array([hypercube.vertices_map[idx].coordinates.copy() for idx in face.vertices_indices])
            for face in self.faces
        ]
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        adjacency = {i: [] for i in range(len(self.faces))}
        required_shared_vertices = 2 ** (self.face_dim - 1)
        for i in range(len(self.faces)):
            for j in range(i + 1, len(self.faces)):
                shared = self.face_vertices_indices[i] & self.face_vertices_indices[j]
                if len(shared) == required_shared_vertices:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        return adjacency

    def iter_spanning_trees(self, max_trees=None):
        edges = []
        for i in range(len(self.faces)):
            for j in self.adjacency[i]:
                if i < j:
                    edges.append((i, j))
        edges.sort()

        n = len(self.faces)
        target_edges = n - 1

        def backtrack(edge_idx, dsu, chosen):
            if max_trees is not None and len(results) >= max_trees:
                return
            if len(chosen) > target_edges:
                return
            if edge_idx == len(edges):
                if len(chosen) == target_edges and dsu.count == 1:
                    results.append(chosen[:])
                return

            a, b = edges[edge_idx]

            # Option 1: include this edge if it doesn't create a cycle.
            dsu_in = dsu.copy()
            if dsu_in.union(a, b):
                chosen.append((a, b))
                backtrack(edge_idx + 1, dsu_in, chosen)
                chosen.pop()

            # Option 2: exclude this edge.
            backtrack(edge_idx + 1, dsu, chosen)

        results = []
        backtrack(0, _DisjointSet(len(self.faces)), [])
        for tree_edges in results:
            yield tree_edges

    def iter_nets(self, root_face_id=0, max_trees=None, overlap_eps=1e-8):
        for tree_edges in self.iter_spanning_trees(max_trees=max_trees):
            unfolded = self._unfold_tree(tree_edges, root_face_id=root_face_id)
            if unfolded is None:
                continue
            if self._is_non_overlapping(unfolded, eps=overlap_eps):
                yield UnfoldedNet(
                    dimension=self.face_dim,
                    face_vertices=unfolded,
                    tree_edges=tree_edges,
                    root_face_id=root_face_id,
                )

    def _unfold_tree(self, tree_edges, root_face_id=0):
        # Build tree adjacency and parent map from tree_edges.
        tree_adj = {i: [] for i in range(len(self.faces))}
        for a, b in tree_edges:
            tree_adj[a].append(b)
            tree_adj[b].append(a)

        if root_face_id not in tree_adj:
            return None

        parent = {root_face_id: None}
        queue = [root_face_id]
        order = [root_face_id]
        while queue:
            u = queue.pop(0)
            for v in tree_adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                queue.append(v)
                order.append(v)

        if len(parent) != len(self.faces):
            return None

        children = {i: [] for i in range(len(self.faces))}
        for child_id, parent_id in parent.items():
            if parent_id is not None:
                children[parent_id].append(child_id)

        positions_nd = [coords.copy() for coords in self.face_positions_nd]
        center = np.full(self.n, 0.5, dtype=float)

        for face_id in order[1:]:
            parent_id = parent[face_id]
            shared = self.face_vertices_indices[face_id] & self.face_vertices_indices[parent_id]
            shared_coords = self._coords_for_indices(positions_nd[parent_id], shared, parent_id)
            axis_basis = self._compute_basis(shared_coords, self.face_dim - 1)
            comp_basis = self._orthonormal_complement(axis_basis)

            parent_coords = positions_nd[parent_id]
            child_coords = positions_nd[face_id]
            parent_normal = self._face_normal(parent_coords, center)
            child_normal = self._face_normal(child_coords, center)

            angle = self._angle_between_in_plane(comp_basis, child_normal, parent_normal)
            if np.isnan(angle):
                return None

            rotation = self._rotation_in_plane(comp_basis[0], comp_basis[1], angle)
            pivot = shared_coords.mean(axis=0)
            for sub_id in self._iter_subtree(face_id, children):
                positions_nd[sub_id] = self._apply_rotation(positions_nd[sub_id], rotation, pivot)

        root_coords = positions_nd[root_face_id]
        root_origin = root_coords.mean(axis=0)
        root_basis = self._root_axis_basis(root_coords)
        if root_basis is None:
            root_basis = self._compute_basis(root_coords, self.face_dim)
        unfolded = {}
        for face_id, coords in enumerate(positions_nd):
            unfolded[face_id] = np.array([root_basis @ (v - root_origin) for v in coords])
        return unfolded

    def _coords_for_indices(self, face_coords, indices, face_id):
        coords = []
        index_to_vertex = list(self.face_vertices_indices[face_id])
        for idx in indices:
            coords.append(face_coords[index_to_vertex.index(idx)])
        return np.array(coords)

    def _face_normal(self, coords, center):
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=True)
        normal = vt[self.face_dim]
        if np.dot(normal, coords.mean(axis=0) - center) < 0:
            normal = -normal
        return normal / np.linalg.norm(normal)

    def _compute_basis(self, coords, target_dim):
        if target_dim == 0:
            return np.zeros((0, self.n))
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[:target_dim]

    def _orthonormal_complement(self, axis_basis):
        if axis_basis.shape[0] == 0:
            _, _, vt = np.linalg.svd(np.zeros((1, self.n)), full_matrices=True)
            return vt[:2]
        _, _, vt = np.linalg.svd(axis_basis, full_matrices=True)
        return vt[axis_basis.shape[0]:axis_basis.shape[0] + 2]

    def _angle_between_in_plane(self, plane_basis, vec_from, vec_to):
        a = plane_basis @ vec_from
        b = plane_basis @ vec_to
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        a /= na
        b /= nb
        cross = a[0] * b[1] - a[1] * b[0]
        dot = np.clip(a[0] * b[0] + a[1] * b[1], -1.0, 1.0)
        return float(np.arctan2(cross, dot))

    def _rotation_in_plane(self, e1, e2, angle):
        r = np.identity(self.n)
        e1 = e1.reshape(self.n, 1)
        e2 = e2.reshape(self.n, 1)
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        r += sin_theta * (e2 @ e1.T - e1 @ e2.T)
        r += (cos_theta - 1.0) * (e1 @ e1.T + e2 @ e2.T)
        return r

    def _apply_rotation(self, coords, rotation, pivot):
        return np.array([pivot + rotation @ (v - pivot) for v in coords])

    def _iter_subtree(self, face_id, children):
        stack = [face_id]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(children[current])

    def _is_non_overlapping(self, unfolded, eps=1e-8):
        face_ids = list(unfolded.keys())
        bboxes = {}
        for face_id in face_ids:
            coords = unfolded[face_id]
            bboxes[face_id] = (coords.min(axis=0), coords.max(axis=0))

        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                a_min, a_max = bboxes[face_ids[i]]
                b_min, b_max = bboxes[face_ids[j]]
                overlap = True
                for k in range(self.face_dim):
                    if min(a_max[k], b_max[k]) - max(a_min[k], b_min[k]) <= eps:
                        overlap = False
                        break
                if overlap:
                    return False
        return True

    def _root_axis_basis(self, root_coords, tol=1e-8):
        """
        Builds a canonical axis-aligned basis for a facet of a hypercube.
        Returns None if a fixed axis cannot be identified.
        """
        fixed_axis = None
        for axis in range(self.n):
            if np.allclose(root_coords[:, axis], root_coords[0, axis], atol=tol):
                fixed_axis = axis
                break
        if fixed_axis is None:
            return None

        basis = []
        for axis in range(self.n):
            if axis == fixed_axis:
                continue
            e = np.zeros(self.n, dtype=float)
            e[axis] = 1.0
            basis.append(e)
        return np.array(basis, dtype=float)

    @staticmethod
    def net_signature_2d(unfolded_net, tol=1e-6):
        """
        Canonical signature for 2D unfolded nets up to translation/rotation/reflection.
        Works for cube nets (n=3).
        """
        if isinstance(unfolded_net, UnfoldedNet):
            face_vertices = unfolded_net.face_vertices
        else:
            face_vertices = unfolded_net

        points = []
        for coords in face_vertices.values():
            for v in np.array(coords, dtype=float):
                points.append(v)
        if not points:
            return ()

        points = np.array(points, dtype=float)
        points = np.round(points / tol) * tol

        rotations = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, -1.0], [1.0, 0.0]]),
            np.array([[-1.0, 0.0], [0.0, -1.0]]),
            np.array([[0.0, 1.0], [-1.0, 0.0]]),
        ]
        reflect = np.array([[1.0, 0.0], [0.0, -1.0]])

        best = None
        for rot in rotations:
            for is_reflect in (False, True):
                transform = reflect @ rot if is_reflect else rot
                transformed = points @ transform.T
                mins = transformed.min(axis=0)
                normalized = transformed - mins
                normalized = np.round(normalized / tol) * tol
                normalized = normalized[np.lexsort((normalized[:, 1], normalized[:, 0]))]
                signature = tuple(map(tuple, normalized))
                if best is None or signature < best:
                    best = signature

        return best

    @staticmethod
    def net_signature_3d(unfolded_net, tol=1e-6):
        """
        Canonical signature for 3D unfolded nets up to translation/rotation/reflection.
        Works for tesseract nets (n=4).
        """
        if isinstance(unfolded_net, UnfoldedNet):
            face_vertices = unfolded_net.face_vertices
        else:
            face_vertices = unfolded_net

        # Use each cube's minimal corner as a translation-invariant descriptor.
        # This works because unfolded tesseract nets are axis-aligned polycubes.
        cube_mins = []
        for coords in face_vertices.values():
            coords = np.array(coords, dtype=float)
            if coords.shape[1] != 3 or coords.shape[0] < 8:
                continue
            mins = coords.min(axis=0)
            cube_mins.append(mins)
        if not cube_mins:
            return ()

        points = np.array(cube_mins, dtype=float)
        points = np.round(points / tol) * tol

        # Generate all axis permutations with sign flips (48 total).
        transforms = []
        axes = [0, 1, 2]
        for perm in itertools.permutations(axes):
            for signs in itertools.product([-1.0, 1.0], repeat=3):
                mat = np.zeros((3, 3), dtype=float)
                for i, axis in enumerate(perm):
                    mat[i, axis] = signs[i]
                transforms.append(mat)

        best = None
        for mat in transforms:
            transformed = points @ mat.T
            mins = transformed.min(axis=0)
            normalized = transformed - mins
            normalized = np.round(normalized / tol) * tol
            normalized = normalized[np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))]
            signature = tuple(map(tuple, normalized))
            if best is None or signature < best:
                best = signature

        return best
