import argparse

from src.hypercube_factory.exporting.projections import (
    DEFAULT_OUTPUT_DIR,
    make_projection_set,
)
from src.hypercube_factory.exporting.rotations import (
    DEFAULT_ORIENTATION,
    ROTATION_PLANES,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export SCAD files for projections of a rotating tesseract."
    )

    parser.add_argument(
        "-n",
        "--projections",
        type=int,
        default=16,
        help="Number of projection frames to export. Default: 16.",
    )
    parser.add_argument(
        "--projection-type",
        choices=["orthogonal", "perspective"],
        default="perspective",
        help="Projection type. Default: perspective.",
    )
    parser.add_argument(
        "--perspective-distance",
        type=float,
        default=1.5,
        help="Perspective view distance. Used only for perspective projection. Default: 1.5.",
    )
    parser.add_argument(
        "--rotation-plane",
        action="append",
        choices=sorted(ROTATION_PLANES),
        default=None,
        help=(
            "Animated 4D rotation plane. Can be supplied multiple times, "
            "for example, --rotation-plane xw --rotation-plane zw."
            "Default is zw."
        ),
    )
    parser.add_argument(
        "--total-rotation",
        type=float,
        default=90.0,
        help="Total animated rotation in degrees across the exported frames. Default: 90.",
    )
    parser.add_argument(
        "--projection-radius",
        type=float,
        default=0.05,
        help="Radius for projected vertices and edges. Default: 0.05.",
    )

    for plane_name, default_angle in DEFAULT_ORIENTATION.items():
        parser.add_argument(
            f"--base-{plane_name}",
            type=float,
            default=default_angle,
            help=(
                f"Static pre-rotation angle in degrees for the {plane_name.upper()} plane. "
                f"Default: {default_angle}."
            ),
        )

    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Do not translate the tesseract to be centered around the origin before rotation.",
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

    rotation_plane_names = args.rotation_plane or ["zw"]

    init_orientation = {
        "xy": args.base_xy,
        "xz": args.base_xz,
        "xw": args.base_xw,
        "yz": args.base_yz,
        "yw": args.base_yw,
        "zw": args.base_zw,
    }

    print("================================")
    print("= Running Projection Exporter =")
    print("================================")
    print(f"Projections: {args.projections}")
    print(f"Projection type: {args.projection_type}")
    if args.projection_type == "perspective":
        print(f"Perspective distance: {args.perspective_distance}")
    print(f"Animated rotation plane(s): {', '.join(name.upper() for name in rotation_plane_names)}")
    print(f"Total rotation: {args.total_rotation}°")
    print("Base rotations:")
    for plane_name, angle_degrees in init_orientation.items():
        print(f"  {plane_name.upper()}: {angle_degrees}°")

    make_projection_set(
        projection_count=args.projections,
        projection_type=args.projection_type,
        perspective_distance=args.perspective_distance,
        rotation_plane_names=rotation_plane_names,
        total_rotation_degrees=args.total_rotation,
        init_orientation=init_orientation,
        projection_radius=args.projection_radius,
        output_dir=args.output_dir,
        center=not args.no_center,
    )

    print("\n================================")
    print(f"= Export complete. Files saved in '{args.output_dir}'.")
    print("================================")
