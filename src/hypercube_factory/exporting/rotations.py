ROTATION_PLANES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "xw": (0, 3),
    "yz": (1, 2),
    "yw": (1, 3),
    "zw": (2, 3),
}

DEFAULT_ORIENTATION = {
    "xy": 0.0,
    "xz": 0.0,
    "xw": 0.0,
    "yz": 0.0,
    "yw": 0.0,
    "zw": 0.0,
}


def parse_rotation_planes(plane_names):
    """Converts plane names like ['xw', 'zw'] into coordinate index pairs."""
    unknown_planes = sorted(set(plane_names) - set(ROTATION_PLANES))
    if unknown_planes:
        valid_names = ", ".join(sorted(ROTATION_PLANES))
        raise ValueError(
            f"Unknown rotation plane(s): {', '.join(unknown_planes)}. "
            f"Valid planes: {valid_names}."
        )

    return [ROTATION_PLANES[plane_name] for plane_name in plane_names]
