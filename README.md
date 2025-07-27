# Hypercube: A Really Dope Zoetrope

This repository contains the full source code and design files for *Hypercube: A Really Dope Zoetrope*, a kinematic art 
installation that illustrates lower-dimensional representations of a hypercube. The project combines mathematics, 3D modeling 
and printing, manual and CNC woodworking, and microcontroller integration to create a stroboscopic 3D zoetrope.

This installation is being developed for the Mathematical Art Exhibit at the 2027 Joint Mathematics Meeting. It serves as a tool 
for both artistic expression and mathematical visualization, exploring concepts like projection, cross-sections, and unfolding 
through a tangible, animated medium.

## Key Features

- **Mathematical Modeling:** A robust Python `Hypercube` class for generating $n$-dimensional cubes and their constituent 
$k$-faces (vertices, edges, faces, etc.) using principles of graph theory.
- **Geometric Transformations:** Methods for performing rotations, translations, and both orthogonal and perspective projections 
as a means of dimension reduction.
- **Cross-Sections:** Functionality to calculate the cross-sectional shape created when slicing the hypercube with a hyperplane.
- **Nets:** A method for constructing the net of a hypercube by specifying a sequence of $k$-faces to disassociate.  
- **3D Model Generation:** Scripts to export the resulting 3D geometric forms as `.scad` files, ready for import to OpenSCAD.
- **Project Website:** A Jekyll-based site for documenting progress, sharing blog posts, and explaining the core mathematical and 
technical concepts.

## Directory Structure

The repository is organized to separate the core logic from generative scripts, experiments, and output files.

```bash
.
├── src/                # Core Python source code
│   └── hypercube/      # Main package for the project
│       ├── geometry/   # Defines the Hypercube class (structure.py)
│       ├── exporters/  # Scripts for exporting data (e.g., to SCAD)
│       └── ...
├── scripts/            # Standalone Python scripts for generating assets
│   └── generators/     # e.g., generate_scad.py to create 3D models
├── notebooks/          # Jupyter notebooks for experimentation and visualization
├── output/             # Default location for generated files (e.g., .scad, .stl)
├── tests/              # Unit and integration tests for the Python code
├── website/            # Source for the Jekyll-based project website
└── ...                 # Configuration files (Pipfile, LICENSE, etc.)

```

## Installation and Usage

To run the Python scripts and generate your own 3D-printable models, you'll need to set up a local environment first. 

### System Requirements

- [Git](https://git-scm.com/downloads) (optional, but recommended)
- [Python 3.8+](https://www.python.org/downloads/)
- [Pipenv](https://pipenv.pypa.io/en/latest/installation.html) (for managing Python dependencies within a virtual environment)
- [OpenSCAD](https://openscad.org/downloads.html) (required for generating 3D models)

### Local Installation

1. Clone this repository:
   - If using Git, open the terminal `cd` into the directory where you'd like to save the repo, and then run: 
   ```bash
   git clone https://github.com/kristen/hypercube.git
   cd hypercube
   ```
   - Otherwise, download the project's source code and then unzip it in a local directory.
2. Install Python dependencies using Pipenv (which references the included `Pipfile`): 
   ```bash
   pipenv install
   ```
3. Activate the virtual environment:
   ```bash
   pipenv shell
   ```
### Sample Usage

Under construction...

## Contributions

While this project is being developed in the open for transparency and open access, it is currently managed by a small team 
working towards the specific goal. Because of this specific focus, we are not actively seeking pull requests at this time.

We still encourage you to fork the repository for your own experiments. If you have questions, ideas, or discover a bug, 
please feel free to open an issue for discussion. We appreciate all feedback and interest in the project.

## Acknowledgements

This project is partially supported by the **Pathways to STEM Success** research program at San Diego Miramar College, 
which is funded by a grant from the National Science Foundation (NSF).

## License

This project is licenced under the terms of the GPL-3.0 License. See the [`LICENSE`](LICENSE) file for details.