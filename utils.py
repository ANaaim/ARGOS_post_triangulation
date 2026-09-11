from scipy.signal import butter, sosfiltfilt
import numpy as np
import ezc3d
from pathlib import Path


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


def resample_point_data(point_data: np.ndarray, original_frame_rate: float, target_frame_rate: float):
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
    resample_factor = int(target_frame_rate / original_frame_rate)

    # Downsample matrix by keeping every N-th frame along the time axis (axis 2)
    resampled_data = point_data[:, :, ::resample_factor]

    return resampled_data


def filter_point_data_with_NaN(point_data, frame_rate, cutoff=6.0, order=4):
    """
    Applies a zero-phase low-pass Butterworth filter to 3D point data.

    Parameters:
    -----------
    point_data : np.ndarray
        Array of shape (n_channels, n_points, n_frames).
    frame_rate : float or int
        Sampling frequency of the trial (Hz).
    cutoff : float
        Cutoff frequency in Hz (default: 6.0 Hz).
    order : int
        Filter order (default: 2, resulting in an effective 4th-order filter after dual-pass).

    Returns:
    --------
    filtered_data : np.ndarray
        Filtered point data matrix of the same shape.
    """

    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * frame_rate

    # Normalize the cutoff frequency
    normalized_cutoff = cutoff / nyquist_freq

    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)

    # Initialize the filtered data array with NaNs
    filtered_data = np.full_like(point_data, np.nan)

    # Apply the filter to each channel and point, handling NaNs
    for channel in range(point_data.shape[0]):
        for point in range(point_data.shape[1]):
            segment = point_data[channel, point, :]
            valid_indices = ~np.isnan(segment)
            if np.sum(valid_indices) > 3 * max(len(a), len(b)):
                filtered_segment = sosfiltfilt(b, a, segment[valid_indices])
                filtered_data[channel, point, valid_indices] = filtered_segment

    return filtered_data


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

    return to_trim, start_idx, end_idx


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
    c3d["parameters"]["POINT"]["RATE"]["value"] = fq_new_file
    c3d["parameters"]["POINT"]["LABELS"]["value"] = name_points
    c3d["data"]["points"] = points

    # Save the new C3D file
    c3d.write(str(output_path))
