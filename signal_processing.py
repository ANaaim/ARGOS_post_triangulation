import numpy as np
import ezc3d
from pathlib import Path
import pandas as pd

from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt


def transforms_zero_to_nan(point_data: np.ndarray):
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
    resample_factor = int(original_frame_rate / target_frame_rate)

    # Downsample matrix by keeping every N-th frame along the time axis (axis 2)
    resampled_data = point_data[:, :, ::resample_factor]

    return resampled_data


def extract_trim_from_marker_based(path_c3d: Path):
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


def filter_point_data_with_nan_segments(
    points: np.ndarray,
    fs: float,
    cutoff: float = 6.0,
    order: int = 4,
    max_gap: int = 10,
) -> np.ndarray:
    """
    Interpolate short NaN gaps and low-pass filter marker trajectories.

    Parameters
    ----------
    points : np.ndarray
        Marker trajectories with shape (4, n_markers, n_frames).
    fs : float
        Sampling frequency in Hz.
    cutoff : float, optional
        Butterworth low-pass cutoff frequency in Hz.
    order : int, optional
        Butterworth filter order.
    max_gap : int, optional
        Maximum NaN gap length, in frames, to interpolate.

    Returns
    -------
    np.ndarray
        Filtered marker trajectories. Long gaps remain NaN.
    """

    filtered = points.copy()

    sos = butter(
        order,
        cutoff,
        btype="low",
        fs=fs,
        output="sos",
    )

    n_markers = points.shape[1]
    n_frames = points.shape[2]

    for marker in range(n_markers):

        for dim in range(3):

            x = points[dim, marker, :].astype(float)
            x_interp = x.copy()

            # --------------------------------------------------
            # Find NaN segments
            # --------------------------------------------------
            is_nan = np.isnan(x)

            changes = np.diff(np.r_[False, is_nan, False].astype(int))

            gap_starts = np.where(changes == 1)[0]
            gap_ends = np.where(changes == -1)[0]

            # --------------------------------------------------
            # Interpolate only short internal gaps
            # --------------------------------------------------
            for start, end in zip(gap_starts, gap_ends):

                gap_length = end - start

                # Ignore gaps that are too long
                if gap_length > max_gap:
                    continue

                # Cannot interpolate before first valid point
                if start == 0:
                    continue

                # Cannot interpolate after last valid point
                if end >= n_frames:
                    continue

                # Values immediately before and after the gap
                x_known = np.array([start - 1, end])
                y_known = np.array(
                    [
                        x[start - 1],
                        x[end],
                    ]
                )

                # Safety check
                if np.isnan(y_known).any():
                    continue

                interpolator = interp1d(
                    x_known,
                    y_known,
                    kind="linear",
                )

                frames_to_fill = np.arange(start, end)

                x_interp[frames_to_fill] = interpolator(frames_to_fill)

            # --------------------------------------------------
            # Find continuous valid segments after interpolation
            # --------------------------------------------------
            valid = ~np.isnan(x_interp)

            changes = np.diff(np.r_[False, valid, False].astype(int))

            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            y = np.full_like(x_interp, np.nan)

            # --------------------------------------------------
            # Filter each continuous valid segment independently
            # --------------------------------------------------
            for start, end in zip(starts, ends):

                segment = x_interp[start:end]

                padlen = 3 * (2 * len(sos) + 1)

                # Segment too short for filtfilt
                if len(segment) < padlen + 1:
                    y[start:end] = segment
                    continue

                y[start:end] = sosfiltfilt(
                    sos,
                    segment,
                )

            filtered[dim, marker, :] = y

    return filtered
