# function to test if the data are coherent between the marker-based and the markerless c3d files
from pathlib import Path
import numpy as np
from utils import load_c3d


def check_data_presence(path_marker_based: Path, path_marker_less: Path):
    """
    The aim of this test is to verify and identify which files are not both present in the marker-based and markerless folders. This is important to ensure that the data is complete and consistent for further analysis. The test will check for the presence of corresponding files in both folders and report any discrepancies.
    """
    # get all subject folders in the marker-based folder
    subjects_marker_based = [
        d for d in path_marker_based.iterdir() if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    ]
    subjects_marker_less = [
        d for d in path_marker_less.iterdir() if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    ]
    # get all trial in marker-based folder
    trials_marker_based = dict()
    for subject in subjects_marker_based:
        trials_marker_based[subject.name] = [f.stem for f in (subject).glob("*.c3d")]

    # get all trial in markerless folder
    trials_marker_less = dict()
    for subject in subjects_marker_less:
        trials_marker_less[subject.name] = [f.stem for f in (subject).glob("*.c3d")]

    # Compare the trials in both folders and report any discrepancies
    discrepancies = dict()
    for subject in subjects_marker_based:
        subject_name = subject.name

        trials_mb = set(trials_marker_based.get(subject_name, []))
        trials_ml = set(trials_marker_less.get(subject_name, []))
        # print(trials_mb)
        # print(trials_ml)
        missing_in_mb = trials_ml - trials_mb
        missing_in_ml = trials_mb - trials_ml

        if missing_in_mb or missing_in_ml:
            discrepancies[subject_name] = {
                "missing_in_marker_based": list(missing_in_mb),
                "missing_in_marker_less": list(missing_in_ml),
            }

    return discrepancies


def check_frame_numbers(path_marker_based: Path, path_marker_less: Path):
    """
    Check if the frame numbers of the marker-based and markerless c3d files are coherent.
    """
    # get all subject folders in the marker-based folder
    subjects_marker_based = [
        d for d in path_marker_based.iterdir() if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    ]
    subjects_marker_less = [
        d for d in path_marker_less.iterdir() if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    ]

    # Compare the frame numbers in both folders and report any discrepancies
    discrepancies = dict()
    for subject in subjects_marker_based:
        subject_name = subject.name

        trials_mb = {f.stem: f for f in (subject).glob("*.c3d")}
        trials_ml = {f.stem: f for f in (path_marker_less / subject_name).glob("*.c3d")}

        for trial_name, mb_file in trials_mb.items():
            ml_file = trials_ml.get(trial_name)
            if ml_file:
                # Load C3D files and compare frame numbers
                mb_c3d, points_mb, fq_mb, labels_mb = load_c3d(mb_file)
                ml_c3d, points_ml, fq_ml, labels_ml = load_c3d(ml_file)
                mb_frames = points_mb.shape[2]
                ml_frames = points_ml.shape[2]
                print(f"{mb_frames},  {ml_frames} MB vs ML: Subject: {subject_name}, Trial: {trial_name},")
                if mb_frames != ml_frames:
                    discrepancies.setdefault(subject_name, []).append(
                        {"trial": trial_name, "marker_based_frames": mb_frames, "marker_less_frames": ml_frames}
                    )

    return discrepancies


def generate_missing_point_NaN_in_files(path_data: Path, is_marker_less: bool = True):
    # For one file generate all the segment where data are NaN and create a log file with the info
    for subject in path_data.iterdir():
        if not subject.is_dir():
            continue
        subject_name = subject.name
        if is_marker_less:
            trials = {f.stem: f for f in (subject).glob("*.c3d")}
        else:
            trials = {f.stem: f for f in (subject).glob("*.c3d")}
        # Save the missing segments to a log file
        log_file_path = subject / "missing_points_log.txt"
        # Add the missing segments to the log file
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Missing Points Log for {subject_name}\n")
            for trial_name, trial_file in trials.items():
                log_file.write(f"\nTrial: {trial_name}\n")
                c3d, points, fq, labels = load_c3d(trial_file)
                n_frames = points.shape[2]
                missing_segments = []
                current_segment = None
                for frame_idx in range(n_frames):
                    frame_data = points[:, :, frame_idx]
                    if np.isnan(frame_data).any():
                        if current_segment is None:
                            current_segment = {"start_frame": frame_idx + 1}  # 1-based index
                    else:
                        if current_segment is not None:
                            current_segment["end_frame"] = frame_idx  # 1-based index
                            missing_segments.append(current_segment)
                            current_segment = None
                if current_segment is not None:
                    current_segment["end_frame"] = n_frames  # 1-based index
                    missing_segments.append(current_segment)

                if missing_segments:
                    print(f"Logging missing points segments for {trial_name} in {log_file_path}")
                    log_file.write(f"Missing Points Segments for {trial_name}:\n")
                    for segment in missing_segments:
                        log_file.write(f"Start Frame: {segment['start_frame']}, End Frame: {segment['end_frame']}\n")

if __name__ == "__main__":
    # Example usage
    path_marker_less = Path(".\\data\\pre_processed\\marker_less\\RTMPose")
    path_marker_based = Path(".\\data\\pre_processed\\marker_based")
    discrepancies_presence = check_data_presence(path_marker_based, path_marker_less)
    discrepancies_frames = check_frame_numbers(path_marker_based, path_marker_less)

    generate_missing_point_NaN_in_files(path_marker_based, is_marker_less=False)
    generate_missing_point_NaN_in_files(path_marker_less, is_marker_less=True)


    discrepancies = {**discrepancies_presence, **discrepancies_frames}

    if discrepancies:
        print("Discrepancies found:")
        for subject, issues in discrepancies.items():
            print(f"Subject: {subject}")
            if issues["missing_in_marker_based"]:
                print(f"  Missing in Marker-Based: {issues['missing_in_marker_based']}")
            if issues["missing_in_marker_less"]:
                print(f"  Missing in Marker-Less: {issues['missing_in_marker_less']}")
    else:
        print("No discrepancies found. All files are present in both folders.")
