from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hypercube_factory.geometry.structure import Hypercube, HypercubeNetEnumerator
from src.hypercube_factory.graphics.plot_3d import net_traces_3d

# --- Plotly configuration --------------------
import plotly.io as pio
pio.renderers.default = "browser"
# ---------------------------------------------

import plotly.graph_objects as go
import json


def main():
    """Enumerate unique tesseract nets and build a single-page viewer."""
    tesseract = Hypercube(4)
    enumerator = HypercubeNetEnumerator(tesseract)
    palette = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
    ]

    unique_nets = []
    seen_signatures = set()
    total_nets = 0

    for net in enumerator.iter_nets(max_trees=None):
        total_nets += 1
        signature = enumerator.net_signature_3d(net)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_nets.append(net)
        # 261 is the known count of unique 4D hypercube nets up to symmetry.
        if len(unique_nets) >= 261:
            break

    if not unique_nets:
        print("No unique nets found.")
        return

    # Each unique net becomes a Plotly frame for quick Prev/Next navigation.
    frames = []
    for idx, net in enumerate(unique_nets):
        traces = net_traces_3d(net, face_palette=palette)
        frames.append(go.Frame(data=traces, name=str(idx)))

    fig = go.Figure(
        data=net_traces_3d(unique_nets[0], face_palette=palette),
        frames=frames,
    )

    fig.update_layout(
        title_text="4D Tesseract Nets (Unique, Unfolded in 3D)",
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
    )

    # Serialize to HTML so we can inject lightweight Prev/Next controls.
    fig_dict = fig.to_dict()
    html = _build_html_viewer(fig_dict)
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tesseract_nets_viewer.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"Wrote viewer to {output_path}")
    print(f"Enumerated {total_nets} nets ({len(unique_nets)} unique).")


def _build_html_viewer(fig_dict):
    """Builds a self-contained HTML viewer with Prev/Next controls."""
    data_json = json.dumps(fig_dict["data"])
    layout_json = json.dumps(fig_dict["layout"])
    frames_json = json.dumps(fig_dict.get("frames", []))

    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Tesseract Nets Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
      body {{
        margin: 0;
        font-family: sans-serif;
        background: white;
      }}
      #controls {{
        display: flex;
        gap: 12px;
        padding: 12px;
        align-items: center;
      }}
      #plot {{
        width: 100vw;
        height: calc(100vh - 60px);
      }}
      button {{
        padding: 6px 12px;
        font-size: 14px;
      }}
    </style>
  </head>
  <body>
    <div id="controls">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <span id="label">Net 1</span>
    </div>
    <div id="plot"></div>
    <script>
      const data = {data_json};
      const layout = {layout_json};
      const frames = {frames_json};
      const plot = document.getElementById("plot");
      const label = document.getElementById("label");
      let current = 0;

      Plotly.newPlot(plot, data, layout).then(() => {{
        if (frames.length > 0) {{
          Plotly.addFrames(plot, frames);
        }}
      }});

      function show(index) {{
        if (frames.length === 0) {{
          return;
        }}
        current = (index + frames.length) % frames.length;
        label.textContent = `Net ${{current + 1}}`;
        Plotly.animate(
          plot,
          [frames[current].name],
          {{
            mode: "immediate",
            frame: {{duration: 0, redraw: true}},
            transition: {{duration: 0}}
          }}
        );
      }}

      document.getElementById("prev").addEventListener("click", () => show(current - 1));
      document.getElementById("next").addEventListener("click", () => show(current + 1));
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    main()
