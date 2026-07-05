"""Plot 3D hypercube projections, cross-sections, and nets with Plotly."""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def _pad_coords(coords, axis_indices=(0, 1, 2)):
    coords = np.array(coords, dtype=float).ravel()
    max_axis = max(axis_indices)
    if coords.shape[0] <= max_axis:
        padded = np.zeros(max_axis + 1, dtype=float)
        padded[:coords.shape[0]] = coords
        coords = padded
    return coords[list(axis_indices)]


def _order_face_vertices(face_vertices):
    if len(face_vertices) <= 2:
        return list(range(len(face_vertices)))
    centroid = face_vertices.mean(axis=0)
    centered = face_vertices - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis_2d = vt[:2]
    projected = centered @ basis_2d.T
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    return list(np.argsort(angles))


def _triangulate_faces(face_vertices_list):
    xs = []
    ys = []
    zs = []
    tri_i = []
    tri_j = []
    tri_k = []
    vertex_offset = 0

    for face_vertices in face_vertices_list:
        if len(face_vertices) < 3:
            continue
        order = _order_face_vertices(face_vertices)
        ordered = face_vertices[order]
        for point in ordered:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        for idx in range(1, len(ordered) - 1):
            tri_i.append(vertex_offset + 0)
            tri_j.append(vertex_offset + idx)
            tri_k.append(vertex_offset + idx + 1)
        vertex_offset += len(ordered)

    return xs, ys, zs, tri_i, tri_j, tri_k


def _triangulate_convex_polyhedron(points, vertex_offset):
    xs = []
    ys = []
    zs = []
    tri_i = []
    tri_j = []
    tri_k = []
    hull = ConvexHull(points)
    for tri in hull.simplices:
        p0, p1, p2 = points[tri]
        xs.extend([p0[0], p1[0], p2[0]])
        ys.extend([p0[1], p1[1], p2[1]])
        zs.extend([p0[2], p1[2], p2[2]])
        tri_i.append(vertex_offset + 0)
        tri_j.append(vertex_offset + 1)
        tri_k.append(vertex_offset + 2)
        vertex_offset += 3
    return xs, ys, zs, tri_i, tri_j, tri_k, vertex_offset, hull


def plot_hypercube_3d(
    vertices_dict,
    edges_list,
    faces_list=None,
    title="Hypercube Projection",
    face_color="orange",
    face_opacity=0.35,
    axis_indices=(0, 1, 2),
):
    """
    Plots the vertices and edges of a 3D representation of the hypercube
    using Plotly's scatter3d.

    Args:
        vertices_dict (dict): A dictionary mapping vertex initial_indices to Vertex objects
                              (e.g., from Hypercube.vertices_map or the result of project()).
                              These Vertex objects must have a 'coordinates' attribute that is at least 3D.
        edges_list (list): A list of (vertex_idx1, vertex_idx2) tuples representing edges.
        title (str): The title for the Plotly plot.
    """
    if not vertices_dict:
        print("No vertices to plot.")
        return

    def get_padded_coords(vertex):
        return _pad_coords(vertex.coordinates, axis_indices=axis_indices)

    # Extract coordinates for Plotly.
    x_coords = [get_padded_coords(v)[0] for v in vertices_dict.values()]
    y_coords = [get_padded_coords(v)[1] for v in vertices_dict.values()]
    z_coords = [get_padded_coords(v)[2] for v in vertices_dict.values()]

    # Prepare edge data for Plotly lines.
    edge_x = []
    edge_y = []
    edge_z = []
    for idx1, idx2 in edges_list:
        # Ensure both vertices exist in the current vertices_dict (important for projections)
        if idx1 in vertices_dict and idx2 in vertices_dict:
            v1_x, v1_y, v1_z = get_padded_coords(vertices_dict[idx1])
            v2_x, v2_y, v2_z = get_padded_coords(vertices_dict[idx2])
            edge_x.extend([v1_x, v2_x, None])
            edge_y.extend([v1_y, v2_y, None])
            edge_z.extend([v1_z, v2_z, None])

    # Create Plotly traces.
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Edges'
    )

    trace_vertices = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Vertices',
        text=[f"V{idx}" for idx in vertices_dict.keys()],  # Show vertex index on hover.
        hoverinfo='text'
    )

    traces = [trace_edges, trace_vertices]
    if faces_list:
        face_vertices_list = []
        for face in faces_list:
            coords = [
                get_padded_coords(vertices_dict[idx])
                for idx in face
                if idx in vertices_dict
            ]
            if len(coords) >= 3:
                face_vertices_list.append(np.array(coords, dtype=float))
        if face_vertices_list:
            xs, ys, zs, tri_i, tri_j, tri_k = _triangulate_faces(face_vertices_list)
            trace_faces = go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=tri_i, j=tri_j, k=tri_k,
                color=face_color,
                opacity=face_opacity,
                name='Faces',
                flatshading=True,
                showscale=False,
            )
            traces.insert(0, trace_faces)

    # Create the figure and add traces.
    fig = go.Figure(data=traces)

    # Update layout for better visualization.
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', autorange="reversed"),  # Reverse X for typical math visualization.
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'  # Ensures equal scaling on all axes.
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def plot_cross_section_3d(
    cross_section_vertices,
    cross_section_faces=None,
    title="Hypercube Cross-Section",
    face_color="orange",
    face_opacity=0.35,
):
    if cross_section_vertices is None or len(cross_section_vertices) == 0:
        print("No cross-section vertices to plot.")
        return

    vertices = np.array(cross_section_vertices, dtype=float)
    if vertices.shape[1] < 3:
        padded = np.zeros((vertices.shape[0], 3), dtype=float)
        padded[:, :vertices.shape[1]] = vertices
        vertices = padded

    x_coords = vertices[:, 0].tolist()
    y_coords = vertices[:, 1].tolist()
    z_coords = vertices[:, 2].tolist()

    traces = []

    if cross_section_faces is not None and len(cross_section_faces) > 0:
        faces = np.array(cross_section_faces, dtype=int)
        trace_faces = go.Mesh3d(
            x=x_coords, y=y_coords, z=z_coords,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=face_color,
            opacity=face_opacity,
            name='Faces',
            flatshading=True,
            showscale=False,
        )
        traces.append(trace_faces)

        edge_x = []
        edge_y = []
        edge_z = []
        for tri in faces:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edge_x.extend([x_coords[tri[a]], x_coords[tri[b]], None])
                edge_y.extend([y_coords[tri[a]], y_coords[tri[b]], None])
                edge_z.extend([z_coords[tri[a]], z_coords[tri[b]], None])
        traces.append(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Edges'
            )
        )

    traces.append(
        go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(size=5, color='red', symbol='circle'),
            name='Vertices'
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', autorange="reversed"),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_hypercube_net_3d(
    net,
    title="Hypercube Net",
    face_color="orange",
    face_opacity=0.35,
    axis_indices=(0, 1, 2),
):
    if net is None or not getattr(net, "faces", None):
        print("No net faces to plot.")
        return

    face_vertices_list = []
    edge_x = []
    edge_y = []
    edge_z = []
    vertex_coords = []

    cube_faces = [
        (0, 1, 3, 2),  # z = 0
        (4, 5, 7, 6),  # z = 1
        (0, 1, 5, 4),  # y = 0
        (2, 3, 7, 6),  # y = 1
        (0, 2, 6, 4),  # x = 0
        (1, 3, 7, 5),  # x = 1
    ]

    for face in net.faces.values():
        coords_map = {
            idx: _pad_coords(face.vertex_positions[idx], axis_indices=axis_indices)
            for idx in face.vertices_indices
        }
        if len(coords_map) < 2:
            continue

        if face.dimension == 3 and len(coords_map) == 8 and set(coords_map.keys()) == set(range(8)):
            for quad in cube_faces:
                quad_coords = np.array([coords_map[i] for i in quad], dtype=float)
                face_vertices_list.append(quad_coords)
                vertex_coords.extend(quad_coords.tolist())
                for i in range(len(quad_coords)):
                    j = (i + 1) % len(quad_coords)
                    edge_x.extend([quad_coords[i][0], quad_coords[j][0], None])
                    edge_y.extend([quad_coords[i][1], quad_coords[j][1], None])
                    edge_z.extend([quad_coords[i][2], quad_coords[j][2], None])
            continue

        coords_array = np.array(list(coords_map.values()), dtype=float)
        face_vertices_list.append(coords_array)
        order = _order_face_vertices(coords_array)
        ordered = coords_array[order]
        vertex_coords.extend(ordered.tolist())
        for i in range(len(ordered)):
            j = (i + 1) % len(ordered)
            edge_x.extend([ordered[i][0], ordered[j][0], None])
            edge_y.extend([ordered[i][1], ordered[j][1], None])
            edge_z.extend([ordered[i][2], ordered[j][2], None])

    traces = []
    if face_vertices_list:
        xs, ys, zs, tri_i, tri_j, tri_k = _triangulate_faces(face_vertices_list)
        traces.append(
            go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=tri_i, j=tri_j, k=tri_k,
                color=face_color,
                opacity=face_opacity,
                name='Faces',
                flatshading=True,
                showscale=False,
            )
        )

    if edge_x:
        traces.append(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Edges'
            )
        )

    if vertex_coords:
        vertices = np.array(vertex_coords, dtype=float)
        traces.append(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', symbol='circle'),
                name='Vertices'
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', autorange="reversed"),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_hypercube_net_2d(
    unfolded_net,
    title="Hypercube Net (Unfolded)",
    face_color="rgba(255,165,0,0.35)",
    edge_color="blue",
    axis_equal=True,
    face_palette=None,
):
    if unfolded_net is None or not getattr(unfolded_net, "face_vertices", None):
        print("No unfolded net faces to plot.")
        return

    traces = net_traces_2d(
        unfolded_net,
        face_color=face_color,
        edge_color=edge_color,
        face_palette=face_palette,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1) if axis_equal else dict(),
        yaxis=dict(),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )
    fig.show()


def net_traces_2d(
    unfolded_net,
    face_color="rgba(255,165,0,0.35)",
    edge_color="blue",
    face_palette=None,
):
    """Return Plotly traces for 2D unfolded nets (squares in a plane)."""
    if unfolded_net is None or not getattr(unfolded_net, "face_vertices", None):
        return []

    traces = []
    face_items = sorted(unfolded_net.face_vertices.items(), key=lambda item: item[0])
    for face_id, coords in face_items:
        coords = np.array(coords, dtype=float)
        if coords.shape[1] != 2 or len(coords) < 3:
            continue
        order = _order_face_vertices(coords)
        ordered = coords[order]
        x = ordered[:, 0].tolist()
        y = ordered[:, 1].tolist()
        x.append(ordered[0, 0])
        y.append(ordered[0, 1])
        color = face_color
        if face_palette:
            color = face_palette[face_id % len(face_palette)]
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=edge_color, width=2),
                fill="toself",
                fillcolor=color,
                name=f"Face {face_id}",
                showlegend=False,
            )
        )
    return traces


def net_traces_3d(
    unfolded_net,
    face_color="rgba(255,165,0,0.25)",
    edge_color="blue",
    edge_width=2,
    face_palette=None,
):
    """Return Plotly traces for 3D unfolded nets (axis-aligned cubes)."""
    if unfolded_net is None or not getattr(unfolded_net, "face_vertices", None):
        return []

    face_vertices_list = []
    face_colors = []
    edge_x = []
    edge_y = []
    edge_z = []

    face_items = sorted(unfolded_net.face_vertices.items(), key=lambda item: item[0])
    for face_id, coords in face_items:
        coords = np.array(coords, dtype=float)
        if coords.shape[1] != 3 or len(coords) < 8:
            continue
        # Each cube face is the set of vertices with one coordinate fixed.
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        faces = []
        for axis in range(3):
            for val in (mins[axis], maxs[axis]):
                face = coords[np.isclose(coords[:, axis], val)]
                if len(face) == 4:
                    faces.append(face)
        for face in faces:
            face_vertices_list.append(face)
            face_colors.append(face_palette[face_id % len(face_palette)] if face_palette else face_color)
            order = _order_face_vertices(face)
            ordered = face[order]
            for i in range(len(ordered)):
                j = (i + 1) % len(ordered)
                edge_x.extend([ordered[i][0], ordered[j][0], None])
                edge_y.extend([ordered[i][1], ordered[j][1], None])
                edge_z.extend([ordered[i][2], ordered[j][2], None])

    traces = []
    if face_vertices_list:
        xs, ys, zs, tri_i, tri_j, tri_k = _triangulate_faces(face_vertices_list)
        # Apply per-face colors by building a color list per triangle.
        facecolor = []
        tri_count = 0
        for color in face_colors:
            facecolor.extend([color, color])
            tri_count += 2
        facecolor = facecolor[:len(tri_i)]
        traces.append(
            go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=tri_i, j=tri_j, k=tri_k,
                facecolor=facecolor,
                opacity=0.6,
                name="Faces",
                flatshading=True,
                showscale=False,
            )
        )

    if edge_x:
        traces.append(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                name="Edges",
                showlegend=False,
            )
        )

    return traces


def net_traces_3d_from_frame(
    frame,
    face_color="rgba(255,165,0,0.25)",
    edge_color="blue",
    edge_width=2,
    face_palette=None,
):
    """Return Plotly traces for a single folded frame (dict face_id -> vertex map)."""
    if frame is None:
        return []

    xs = []
    ys = []
    zs = []
    tri_i = []
    tri_j = []
    tri_k = []
    facecolors = []
    edge_x = []
    edge_y = []
    edge_z = []
    vertex_offset = 0
    edge_tol = 1.05

    face_items = sorted(frame.items(), key=lambda item: item[0])
    for face_id, vertex_map in face_items:
        indices = sorted(vertex_map.keys())
        coords = np.array([vertex_map[idx] for idx in indices], dtype=float)
        if coords.size == 0:
            continue
        if coords.shape[1] < 3:
            padded = np.zeros((coords.shape[0], 3), dtype=float)
            padded[:, :coords.shape[1]] = coords
            coords = padded

        color = face_palette[face_id % len(face_palette)] if face_palette else face_color

        if coords.shape[0] == 4:
            order = _order_face_vertices(coords)
            ordered = coords[order]
            for point in ordered:
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
            for idx in range(1, len(ordered) - 1):
                tri_i.append(vertex_offset + 0)
                tri_j.append(vertex_offset + idx)
                tri_k.append(vertex_offset + idx + 1)
                facecolors.append(color)
            for i in range(len(ordered)):
                j = (i + 1) % len(ordered)
                edge_x.extend([ordered[i][0], ordered[j][0], None])
                edge_y.extend([ordered[i][1], ordered[j][1], None])
                edge_z.extend([ordered[i][2], ordered[j][2], None])
            vertex_offset += len(ordered)
            continue

        if coords.shape[0] >= 4:
            try:
                tri_x, tri_y, tri_z, t_i, t_j, t_k, vertex_offset, hull = _triangulate_convex_polyhedron(
                    coords,
                    vertex_offset,
                )
            except Exception:
                continue
            xs.extend(tri_x)
            ys.extend(tri_y)
            zs.extend(tri_z)
            tri_i.extend(t_i)
            tri_j.extend(t_j)
            tri_k.extend(t_k)
            facecolors.extend([color] * len(t_i))
            if coords.shape[0] == 8:
                distances = []
                for i in range(coords.shape[0]):
                    for j in range(i + 1, coords.shape[0]):
                        distances.append(np.linalg.norm(coords[i] - coords[j]))
                min_dist = min(d for d in distances if d > 1e-12)
                threshold = min_dist * edge_tol
                for i in range(coords.shape[0]):
                    for j in range(i + 1, coords.shape[0]):
                        if np.linalg.norm(coords[i] - coords[j]) <= threshold:
                            edge_x.extend([coords[i][0], coords[j][0], None])
                            edge_y.extend([coords[i][1], coords[j][1], None])
                            edge_z.extend([coords[i][2], coords[j][2], None])

    traces = []
    if tri_i:
        traces.append(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=tri_i,
                j=tri_j,
                k=tri_k,
                facecolor=facecolors if facecolors else None,
                color=None if facecolors else face_color,
                opacity=0.6,
                name="Faces",
                flatshading=True,
                showscale=False,
            )
        )

    if edge_x:
        traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                name="Edges",
                showlegend=False,
            )
        )

    return traces


def plot_hypercube_net_frames_3d(
    frames,
    title="Hypercube Net Animation",
    face_color="rgba(255,165,0,0.25)",
    edge_color="blue",
    edge_width=2,
    face_palette=None,
):
    """Plot a slider-controlled animation over a list of frame dicts."""
    if not frames:
        print("No frames to plot.")
        return

    base_traces = net_traces_3d_from_frame(
        frames[0],
        face_color=face_color,
        edge_color=edge_color,
        edge_width=edge_width,
        face_palette=face_palette,
    )

    plot_frames = []
    for idx, frame in enumerate(frames):
        traces = net_traces_3d_from_frame(
            frame,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=edge_width,
            face_palette=face_palette,
        )
        plot_frames.append(go.Frame(data=traces, name=str(idx)))

    steps = []
    for i in range(len(frames)):
        steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                label=str(i + 1),
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frame "},
            pad={"t": 30},
            steps=steps,
        )
    ]

    fig = go.Figure(data=base_traces, frames=plot_frames)
    fig.update_layout(
        title=title,
        showlegend=False,
        sliders=sliders,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()
