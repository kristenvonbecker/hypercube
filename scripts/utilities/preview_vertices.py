import numpy as np
import plotly.graph_objects as go

def plot_hypercube_3d(vertices_dict, edges_list, title="Hypercube Projection"):
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

    # Assuming coordinates are 3D after projection or for a 3D hypercube
    # If a vertex has fewer than 3 dimensions, pad with zeros for plotting
    def get_padded_coords(vertex):
        coords = vertex.coordinates
        if len(coords) >= 3:
            return coords[0], coords[1], coords[2]
        elif len(coords) == 2:
            return coords[0], coords[1], 0.0 # Pad Z with zero for 2D projection
        elif len(coords) == 1:
            return coords[0], 0.0, 0.0 # Pad Y, Z with zeros for 1D projection
        else: # 0D point
            return 0.0, 0.0, 0.0


    # Extract coordinates for Plotly
    x_coords = [get_padded_coords(v)[0] for v in vertices_dict.values()]
    y_coords = [get_padded_coords(v)[1] for v in vertices_dict.values()]
    z_coords = [get_padded_coords(v)[2] for v in vertices_dict.values()]

    # Prepare edge data for Plotly lines
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

    # Create Plotly traces
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
        text=[f"V{idx}" for idx in vertices_dict.keys()], # Show vertex index on hover
        hoverinfo='text'
    )

    # Create the figure and add traces
    fig = go.Figure(data=[trace_edges, trace_vertices])

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', autorange="reversed"), # Reverse X for typical math visualization
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube' # Ensures equal scaling on all axes
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()