from scipy.signal import butter, sosfiltfilt
import numpy as np
import ezc3d
from pathlib import Path
import pandas as pd
from tkinter import Tk, filedialog

def transforms_zero_to_nan(point_data):
    """
    Transforms all [0, 0, 0] points in the point data to NaN.

    Parameters:
    -----------
    point_data : np.ndarray
        Array of shape (n_channels, n_points, n_frames).

    Returns:
    --------
    transformed_data : np.ndarray
        Point data with [0.0, 0.0, 0.0] replaced by NaN.
    """
    transformed_data = np.where(np.all(point_data[0:3, :, :] == 0.0, axis=0), np.nan, point_data)

    return transformed_data

def XYZ_to_ZXY(points_data: np.ndarray):
    """
    Reorders the coordinate system of the point data from (X, Y, Z) to (Z, X, Y) to allow markerless data to beoriented in the same 
    way as the marker-based data.

    Parameters
    ----------
    point : np.ndarray
        Array of shape (4, n_markers, n_frames) representing the point data.
    
    Returns
    -------
    point_data_reoriented : np.ndarray
        Reordered point data with shape (4, n_markers, n_frames).

    """
    # 2. Re-order coordinate system: (X, Y, Z, 1) -> (Z, X, Y, 1)
    point_data_reoriented = points_data[[2, 0, 1, 3], :, :]

    return point_data_reoriented

def resample_point_data(point_data: np.ndarray,
                        original_frame_rate: float,
                        target_frame_rate: float):
    """
    Resamples the point data from the original frame rate to the target frame rate.

    Parameters:
    -----------
    point_data : np.ndarray
        Array of shape (n_channels, n_points, n_frames).
    original_frame_rate : float or int
        Original sampling frequency of the trial (Hz).
    target_frame_rate : float or int
        Desired sampling frequency of the trial (Hz).

    Returns:
    --------
    resampled_data : np.ndarray
        Resampled point data matrix.
    """
    # Calculate the resampling factor (should be an integer)
    resample_factor = int(original_frame_rate/target_frame_rate)

    # Downsample matrix by keeping every N-th frame along the time axis (axis 2)
    resampled_data = point_data[:, :, ::resample_factor]

    return resampled_data

def extract_trim_from_marker_based(path_c3d):
    """Trims 3D point data based on 'begin' and 'end' events, appends the trim

    result log message directly into the shared .txt log file, and returns
    both the cut points and metadata dictionary for JSON export.
    """
    file_c3d = ezc3d.c3d(str(path_c3d))
    point_data = file_c3d["data"]["points"]
    n_frames = point_data.shape[2]
    frame_rate = file_c3d["header"]["points"]["frame_rate"]
    first_frame = file_c3d["header"]["points"]["first_frame"]

    # Check for EVENT group safely in C3D parameters
    if "EVENT" in file_c3d["parameters"]:
        labels_events = file_c3d["parameters"]["EVENT"]["LABELS"]["value"]
        times_events = file_c3d["parameters"]["EVENT"]["TIMES"]["value"]
    else:
        labels_events = []
        times_events = []

    start_idx = 0
    end_idx = n_frames
    # check if the end_idx is even, if not, take the previous even frame (to avoid having a frame with no data)
    if end_idx % 2 != 0:    
        end_idx -= 1
    # to allow synchronisation with data with a half frame rate.     
    has_begin = False
    has_end = False
    begin_frame = None
    end_frame = None

    # Search for 'begin' and 'end' events in the file
    for i, label in enumerate(labels_events):
        label_clean = label.strip().lower()

        # Calculate total seconds from minutes (row 0) and seconds (row 1)
        total_seconds = (times_events[0][i] * 60) + times_events[1][i]

        # Convert timestamp to frame index relative to array start
        frame_number = round(total_seconds * frame_rate)
        frame_idx = int(frame_number - first_frame)

        # Protect against out-of-bounds indices
        frame_idx = max(0, min(frame_idx, n_frames))

        if "begin" in label_clean:
            start_idx = frame_idx
            begin_frame = frame_idx + 1  # 1-based index for human readability
            has_begin = True

        if "end" in label_clean:
            end_idx = frame_idx + 1
            end_frame = frame_idx + 1  # 1-based index for human readability
            has_end = True

    # Sanity check for valid indices
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_frames       
        has_begin = False
        has_end = False
        begin_frame = None
        end_frame = None
    to_trim = has_begin or has_end

    # check if the start_idx is even, if not, take the next even frame (to avoid having a frame with no data)
    if start_idx % 2 != 0:
        start_idx += 1
    # check if the end_idx is even, if not, take the previous even frame (to avoid having a frame with no data)
    if end_idx % 2 != 0:
        end_idx -= 1

    return to_trim, start_idx, end_idx


def load_c3d(path: Path):
    c3d = ezc3d.c3d(str(path))
    return (
        c3d,
        c3d["data"]["points"],
        c3d["parameters"]["POINT"]["RATE"]["value"][0],
        c3d["parameters"]["POINT"]["LABELS"]["value"]
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

def calculate_RAB(point_data: np.ndarray, name_point_list: list):
    # TODO: to make more clear
    # 1. Extract marker labels list from the C3D object
    all_labels = [label.strip() for label in name_point_list]

    n_frames = point_data.shape[2]
    # Helper function to get marker coordinates safely
    def get_marker(label_name):
        return point_data[0:4, all_labels.index(label_name), :]

    # --- 2. CALCULATE VIRTUAL MARKERS ---

    # Midpoints for ISB Thorax
    mid_end = (get_marker("IJ") + get_marker("C7")) / 2.0
    mid_start = (get_marker("PX") + get_marker("T10")) / 2.0

    Z_thorax = mid_end - mid_start
    Z_thorax /= np.linalg.norm(Z_thorax, axis=0)
    # Bi-acromial distance & Glenohumeral Joint Centers (Rab 2002)
    R_AC = get_marker("R_AC")
    L_AC = get_marker("L_AC")
    D_all = np.linalg.norm(R_AC[:3, :] - L_AC[:3, :], axis=0)
    D = np.mean(D_all)


    R_GH = np.ones((4, n_frames))
    L_GH = np.ones((4, n_frames))

    # Right shoulder center
    R_GH = R_AC.copy() - Z_thorax * (0.17 * D)

    # Left shoulder center
    L_GH = L_AC.copy() - Z_thorax * (0.17 * D)

    return R_GH, L_GH


def filter_point_data_with_nan_segments(
    points,
    fs,
    cutoff=6.0,
    order=4,
    max_gap=10,
):
    """
    Parameters
    ----------
    points : ndarray
        Shape (4, n_markers, n_frames)
    fs : float
        Sampling frequency
    cutoff : float
        Butterworth cutoff frequency
    order : int
        Butterworth order
    max_gap : int
        Fill only gaps shorter than this number of frames
    """

    filtered = points.copy()

    sos = butter(
        order,
        cutoff,
        btype="low",
        fs=fs,
        output="sos"
    )

    n_markers = points.shape[1]

    for marker in range(n_markers):

        for dim in range(3):

            x = points[dim, marker, :].astype(float)

            # ------------------------
            # Fill only short gaps
            # ------------------------
            x_interp = (
                pd.Series(x)
                .interpolate(
                    method="linear",
                    limit=max_gap,
                    limit_direction="both"
                )
                .to_numpy()
            )

            # Remaining NaNs = long gaps
            valid = ~np.isnan(x_interp)

            changes = np.diff(
                np.r_[False, valid, False].astype(int)
            )

            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            y = np.full_like(x_interp, np.nan)

            for start, end in zip(starts, ends):

                segment = x_interp[start:end]

                padlen = 3 * (2 * len(sos) + 1)

                if len(segment) < padlen+1:
                    y[start:end] = segment
                    continue

                y[start:end] = sosfiltfilt(
                    sos,
                    segment
                )

            filtered[dim, marker, :] = y

    return filtered

def choose_file():
    root = Tk()
    root.withdraw()  # cache la fenêtre principale

    filename = filedialog.askopenfilename(
        title="Choisir un fichier NPY",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )

    root.destroy()
    return filename
