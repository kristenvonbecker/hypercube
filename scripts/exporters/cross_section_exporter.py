import argparse

from src.hypercube_factory.exporting.cross_sections import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ORIENTATION,
    build_oriented_tesseract,
    get_slice_bounds,
    make_sliceset,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export SCAD files for cross-sections of a rotated tesseract."
    )

    parser.add_argument(
        "-n",
        "--sections",
        type=int,
        default=16,
        help="Number of cross-sections to export. Default: 16.",
    )
    parser.add_argument(
        "--slice-min",
        type=float,
        default=None,
        help="Minimum slice value. If omitted, computed from the rotated hypercube.",
    )
    parser.add_argument(
        "--slice-max",
        type=float,
        default=None,
        help="Maximum slice value. If omitted, computed from the rotated hypercube.",
    )

    for plane_name, angle in DEFAULT_ORIENTATION.items():
        parser.add_argument(
            f"--{plane_name}",
            type=float,
            default=angle,
            help=(
                f"Initial rotation angle in degrees for the {plane_name.upper()} plane. "
                f"Default: {angle}."
            ),
        )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported SCAD files. Default: {DEFAULT_OUTPUT_DIR}.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    rotation_degrees = {
        "xy": args.xy,
        "xz": args.xz,
        "xw": args.xw,
        "yz": args.yz,
        "yw": args.yw,
        "zw": args.zw,
    }

    oriented_tesseract = build_oriented_tesseract(orientation=DEFAULT_ORIENTATION)
    computed_slice_min, computed_slice_max = get_slice_bounds(
        oriented_tesseract,
        slice_dimension=3,
    )

    effective_slice_min = computed_slice_min if args.slice_min is None else args.slice_min
    effective_slice_max = computed_slice_max if args.slice_max is None else args.slice_max

    print("===================================")
    print("= Running Cross-Section Exporter =")
    print("===================================")
    print(f"Sections: {args.sections}")
    print(f"Slice range: {effective_slice_min:.6f} to {effective_slice_max:.6f}")
    print("Rotations:")
    for plane_name, angle_degrees in rotation_degrees.items():
        print(f"  {plane_name.upper()}: {angle_degrees}°")

    make_sliceset(
        section_count=args.sections,
        slice_min=args.slice_min,
        slice_max=args.slice_max,
        orientation=rotation_degrees,
        output_dir=args.output_dir,
    )

    print("\n===================================")
    print(f"= Export complete. Files saved in '{args.output_dir}'.")
    print("===================================")
