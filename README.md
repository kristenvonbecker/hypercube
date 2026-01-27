# Hypercube: A Really Dope Zoetrope

This repository contains the source code and design files for *Hypercube: A Really Dope Zoetrope*, a kinematic art 
   installation that illustrates lower-dimensional representations of a (4-dimensional) [tesseract](https://en.wikipedia.org/wiki/Tesseract).

*Note:* If the words "hypercube" and "zoetrope" make your head spin, check out the [project website](https://kristen.github.io/hypercube/) first/instead.

## Key Features

- **Mathematically-meaningful class design:** The project’s class structure is intentionally aligned with the mathematical definition
of hypercubes, so the software model reflects the conceptual model.
- **Generalized dimensional operations:** A unified toolkit for projections, cross-sections, and nets/unfoldings:
related ways of producing lower-dimensional representations.
- **Transformations beyond reduction:** Geometric transforms (including double-rotation) are used not just for dimension reduction,
but for expressing higher-dimensional motion via the zoetrope.
- **Fabrication outputs:** Export pathways for physical builds (e.g. SCAD files), plus asset generation for both the installation and its...
- **Public-facing explainer site:** A Jekyll website with embedded interactive visuals (e.g. Plotly exports) and progress
documentation.

## Directory Structure

The repository is organized to separate reusable library code, project pipelines, experiments, design assets, and generated artifacts.

```bash
.
├── src/                        # Core Python source code
│   └── hypercube_factory/      # Main package for the project
│       ├── geometry/           # Hypercube structure and geometry
│       ├── graphics/           # Plotting and visualization helpers
│       ├── animations/         # Animation utilities
│       └── scad/               # OpenSCAD export helpers
├── apps/                       # Runnable pipelines (zoetrope + website exports)
│   ├── zoetrope/
│   └── website_assets/
├── experiments/                # Demos + notebooks for exploration
│   ├── demos/
│   └── notebooks/
├── design/                     # Physical design assets + specs
│   ├── cad/
│   ├── specs/
│   ├── renders/
│   ├── references/
│   └── electronics/
├── artifacts/                  # Generated outputs (ignored by git)
│   ├── scad/
│   ├── renders/
│   └── plots/
├── tests/                      # Unit and integration tests
├── website/                    # Jekyll site source
└── ...                         # Config files (Pipfile, LICENSE, etc.)

```

## Installation and Usage

To run the Python code and generate your own 3D-printable models, you'll need to set up a local environment first. 

### System Requirements

- [Git](https://git-scm.com/downloads) (optional, but recommended)
- [Python 3.x](https://www.python.org/downloads/)
- [Pipenv](https://pipenv.pypa.io/en/latest/installation.html) (for managing Python dependencies within a virtual environment)
- [OpenSCAD](https://openscad.org/downloads.html) (required for generating 3D models)

### Local Installation

1. Clone this repository:
   - If using Git, open the terminal `cd` into the directory where you'd like to save the repo, and then run: 
   ```bash
   git clone https://github.com/kristenvonbecker/hypercube.git
   cd hypercube
   ```
   - Otherwise, download the project's source code and then unzip it in a local directory.
3. From the project root, install Python dependencies using Pipenv (which references the included `Pipfile`): 
   ```bash
   pipenv install
   ```
3. Activate the virtual environment:
   ```bash
   pipenv shell
   ```
### Sample Usage

Under construction. See `experiments/demos/` for examples and `apps/` for pipeline entry points.

## Acknowledgements

This project has been partially supported by the **Pathways to STEM Success** research program at San Diego Miramar College, 
which is funded by a grant from the National Science Foundation (NSF).

## License

This project is licenced under the terms of the GPL-3.0 License. See the [`LICENSE`](LICENSE) file for details.
