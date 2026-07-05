from pathlib import Path


def clear_matching_files(output_dir, pattern):
    """Removes previously exported projection frame files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in output_path.glob(pattern):
        if file.is_file():
            file.unlink()