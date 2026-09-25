from pathlib import Path
import numpy as np
import ezc3d


def load_c3d(path: Path):
    c3d = ezc3d.c3d(str(path))
    return (
        c3d,
        c3d["data"]["points"],
        c3d["parameters"]["POINT"]["RATE"]["value"][0],
        c3d["parameters"]["POINT"]["LABELS"]["value"],
    )


def write_new_c3d(points: np.ndarray, name_points: list, fq_new_file: float, output_path: Path):
    """
    Write a new C3D file with the given points, point names, and frame rate.
    points: np.ndarray of shape (n_channels, n_points, n_frames)
    name_points: list of point names
    fq_new_file: new frame rate (Hz)
    output_path: path to save the new C3D file
    """

    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with random data
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [fq_new_file]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = name_points
    c3d["data"]["points"] = points

    # Save the new C3D file
    c3d.write(str(output_path))
