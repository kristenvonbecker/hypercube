from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hypercube_factory.geometry.structure import Hypercube, HypercubeNetEnumerator
from src.hypercube_factory.graphics.plot_3d import net_traces_2d

# --- Plotly configuration --------------------
import plotly.io as pio
pio.renderers.default = "browser"
# ---------------------------------------------
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    """Render all unique cube nets in a single 2D subplot grid."""
    cube = Hypercube(3)
    enumerator = HypercubeNetEnumerator(cube)

    max_trees = None
    max_unique_nets_to_show = 11

    total_nets = 0
    unique_nets = 0
    seen_signatures = set()
    nets_to_plot = []

    for net in enumerator.iter_nets(max_trees=max_trees):
        total_nets += 1
        signature = enumerator.net_signature_2d(net)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_nets += 1

        nets_to_plot.append(net)
        if unique_nets >= max_unique_nets_to_show:
            break

    # Use a shared bounding box so that a unit square is the same size in every subplot.
    global_min = [float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf")]
    for net in nets_to_plot:
        for coords in net.face_vertices.values():
            for v in coords:
                global_min[0] = min(global_min[0], v[0])
                global_min[1] = min(global_min[1], v[1])
                global_max[0] = max(global_max[0], v[0])
                global_max[1] = max(global_max[1], v[1])
    if global_min[0] == float("inf"):
        global_min = [0.0, 0.0]
        global_max = [1.0, 1.0]
    pad = 0.2
    x_range = [global_min[0] - pad, global_max[0] + pad]
    y_range = [global_min[1] - pad, global_max[1] + pad]

    cols = 4
    rows = max(1, (len(nets_to_plot) + cols - 1) // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Net {i + 1}" for i in range(len(nets_to_plot))],
    )
    for idx, net in enumerate(nets_to_plot):
        row = idx // cols + 1
        col = idx % cols + 1
        for trace in net_traces_2d(net):
            fig.add_trace(trace, row=row, col=col)

    for i in range(1, rows * cols + 1):
        axis_id = "" if i == 1 else str(i)
        fig.update_xaxes(
            scaleanchor=f"y{axis_id}",
            scaleratio=1,
            range=x_range,
            visible=False,
            row=(i - 1) // cols + 1,
            col=(i - 1) % cols + 1,
        )
        fig.update_yaxes(
            range=y_range,
            visible=False,
            row=(i - 1) // cols + 1,
            col=(i - 1) % cols + 1,
        )

    fig.update_layout(
        title_text="3D Cube Nets (Unique, Unfolded in 2D)",
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.show()

    print(f"Enumerated {total_nets} nets ({unique_nets} unique).")


if __name__ == "__main__":
    main()
