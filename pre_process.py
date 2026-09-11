import ezc3d    
from pathlib import Path
from functions_batch_mb import (
    filter_point_data,
    prepro_export_mb,
    resample_point_data_mb,
    trim_c3d_trial_mb,
)
from plot_functions import save_c3d_markers_3d_html
import numpy as np
from utils import (transforms_zero_to_nan, 
                   filter_point_data_with_NaN,
                   extract_trim_from_marker_based,
                   write_new_c3d)

def correct_markerless_fq_folder(folder_marker_based: Path, folder_markerless_to_correct: Path, folder_marker_less_to_export: Path):
    """
    Correct the frame rate of marker-less data based on the corresponding marker-based data.
    """

    for subject_folder in folder_marker_based.iterdir():
        if not subject_folder.is_dir():
            continue
        subject_name = subject_folder.name
        print(f"\n--- Processing subject: {subject_name} ---")
        folder_subject_markerbase = folder_marker_based / subject_name / "Final"

        # Check if the subject folder exists in the markerless folder
        ml_subject_folder = folder_markerless_to_correct / subject_name / "Final"
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue
        
        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_marker_less_to_export / subject_name 
        if not output_subject_folder.exists():
            output_subject_folder.mkdir(parents=True, exist_ok=True)
        for trial in folder_subject_markerbase.glob("*.c3d"):
            trial_name = trial.stem
            print(f"Processing trial: {trial_name}")
            # Check if the corresponding markerless trial exists
            ml_trial_path = ml_subject_folder / f"{trial_name}.c3d"
            if not ml_trial_path.exists():
                print(f"❌ No corresponding markerless trial for {trial_name}")
                continue
            correct_markerless_fq_file(trial, ml_trial_path, output_subject_folder)

def correct_markerless_fq_file(trial: Path, ml_trial_path: Path, output_subject_folder: Path):
    trial_name = trial.stem
    # Read the marker-based and markerless c3d files
    c3d_mb = ezc3d.c3d(str(trial))
    c3d_ml = ezc3d.c3d(str(ml_trial_path))
    # get the frame count of the two c3d files
    
    nb_frames_mb = c3d_mb["data"]["points"].shape[2]
    nb_frames_ml = c3d_ml["data"]["points"].shape[2]
    
    #calculate the ratio of the two frame counts
    ratio = nb_frames_mb / nb_frames_ml
    # find if the ratio is close to 1 or 2  
    ratio_rounded = round(ratio)
    print(ratio)
    print(ratio_rounded)
    if abs(ratio - ratio_rounded) > 0.1:
        print(f"❌ The ratio of the frame counts is not close to 1 or 2 for {trial_name}. Ratio: {ratio}")
        return
    # get fq of mb 
    if ratio_rounded == 2:
        c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
        # check if the rotation rate is present in the marker-based c3d file
        if "ROTATION" in c3d_mb["parameters"]:
            c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
    elif ratio_rounded == 1:
        c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
        # check if the rotation rate is present in the marker-based c3d file
        if "ROTATION" in c3d_mb["parameters"]:
            c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
    else:
        print(f"❌ The ratio of the frame counts is not 1 or 2 for {trial_name}. Ratio: {ratio}")
    
    print(str(output_subject_folder / f"{trial_name}.c3d"))
    c3d_ml.write(str(output_subject_folder / f"{trial_name}.c3d"))



def pre_processed_marker_based_folder(marker_based_folder: Path, output_folder: Path):

    for subject_dir in sorted(marker_based_folder.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith("S") and subject_dir.name[1:].isdigit():
            subject_name = subject_dir.name  # Extracts subject ID (e.g., "S09")
            
            header_text = (
                f"=================================================\n"
                f"      Processing Subject: {subject_name}\n"
                f"=================================================\n"
            )
            print(header_text)

            raw_folder = subject_dir / "Final"
            preprocess_output_folder = output_folder/ subject_name 

            # Verify that the 'Final' folder exists before proceeding
            if not raw_folder.exists():
                print(
                    f"Warning: 'Final' folder not found for {subject_name}. Skipping.\n"
                )
                continue

            # Automatically ensure 'Preprocess' folder exists
            preprocess_output_folder.mkdir(parents=True, exist_ok=True)

            # Point to the existing '3D Snapshot' folder inside Preprocess
            snapshot_folder = preprocess_output_folder / "3D Snapshot"

            # Define log file paths INSIDE each subject's Preprocess folder
            log_file_path = (
                preprocess_output_folder / f"Log_Preprocess_MB_{subject_name}.txt"
            )
            json_file_path = (
                preprocess_output_folder / f"Log_Preprocess_MB_{subject_name}.json"
            )

            # Initialize TXT log file with header banner
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(header_text + "\n")

            # Clear old JSON log file if it exists
            if json_file_path.exists():
                json_file_path.unlink()

            # Reset log containers for the current subject
            valid_trials = {}
            nan_trials = []
            json_log_data = {}

            # Find all C3D files in this subject's 'Final' folder
            c3d_files = list(raw_folder.glob("*.c3d"))
            total_files = len(c3d_files)
            print(f"Found {total_files} C3D files for {subject_name}\n")

            # =========================================================================
            #      D. TRIMMING, NaN VERIFICATION & UPPER LIMBS JOINT DETERMINATION  
            # =========================================================================
            for file_path in c3d_files:
                valid_trials, nan_trials, json_log_data = pre_processed_marker_based_files(file_path =file_path, output_folder = preprocess_output_folder, log_file_path = log_file_path,
                            subject_name = subject_name, valid_trials = valid_trials, nan_trials = nan_trials, json_log_data = json_log_data)
                

def pre_processed_marker_based_files(file_path: Path, output_folder: Path, log_file_path: Path,
                            subject_name: str, valid_trials: dict, nan_trials: list, json_log_data: dict): 
    file_name = file_path.name
    print(f"Processing: {file_name}")

    # 1. Read C3D File
    c3d = ezc3d.c3d(str(file_path))

    # 2. Trim and log
    cut_points, trim_info = trim_c3d_trial_mb(c3d, file_name, log_file_path)

    # 3. NaN check (XYZ only)
    xyz_points = cut_points[:3, :, :]
    has_nan = bool(np.isnan(xyz_points).any())

    # 4. Classify & process
    if has_nan:
        nan_trials.append({
            "filename": file_name,
            "filepath": file_path,
            "points": cut_points,
        })
    else:
        valid_trials[file_name] = cut_points

        # Filter (6 Hz)
        filtered_points = filter_point_data(
            c3d, cut_points, cutoff=6.0, order=2
        )

        # Resample (120 Hz -> 60 Hz)
        resampled_points, new_fps = resample_point_data_mb(
            c3d, filtered_points, step=2
        )

        # Virtual points calculation + export new C3D
        prepro_export_mb(
            c3d, resampled_points, file_name, output_folder
        )

        # Construct path to the exported preprocessed file
        exported_c3d_path = (
            output_folder / f"{file_path.stem}_pp.c3d"
        )

        # Path where the HTML interactive figure will be saved inside '3D Snapshot'
        html_save_path = (
            output_folder / f"{file_path.stem}.html"
        )

        # Save interactive Plotly HTML for valid trials
        save_c3d_markers_3d_html(
            c3d_input=str(exported_c3d_path),
            target_frame=10,
            file_label=f"{subject_name} - {file_path.stem}",
            save_path=str(html_save_path),
        )
        # 5. Save metadata
        json_log_data[file_name] = {**trim_info, "has_nan": has_nan}

    return valid_trials, nan_trials, json_log_data



def pre_processed_marker_less_folder(folder_marker_based: Path, folder_markerless_to_correct: Path, folder_marker_less_to_export: Path):

    for subject_folder in folder_marker_based.iterdir():
            if not subject_folder.is_dir():
                continue
            subject_name = subject_folder.name
            print(f"\n--- Processing subject: {subject_name} ---")
            folder_subject_markerbase = folder_marker_based / subject_name / "Final"

            # Check if the subject folder exists in the markerless folder
            ml_subject_folder = folder_markerless_to_correct / subject_name
            if not ml_subject_folder.exists():
                print(f"❌ No markerless data for {subject_name}")
                continue
            
            # Create the output folder for the corrected markerless data
            output_subject_folder = folder_marker_less_to_export / subject_name 
            if not output_subject_folder.exists():
                output_subject_folder.mkdir(parents=True, exist_ok=True)
            for trial in folder_subject_markerbase.glob("*.c3d"):
                trial_name = trial.stem
                print(f"Processing trial: {trial_name}")
                # Check if the corresponding markerless trial exists
                ml_trial_path = ml_subject_folder / f"{trial_name}.c3d"
                mb_trial_path = folder_subject_markerbase / f"{trial_name}.c3d"
                if not ml_trial_path.exists():
                    print(f"❌ No corresponding markerless trial for {trial_name}")
                    continue
                # remove NaN from the markerless trial
                c3d_ml = ezc3d.c3d(str(ml_trial_path))
                points_ml = c3d_ml["data"]["points"]    
                fq_ml = c3d_ml["parameters"]["POINT"]["RATE"]["value"]

                # Remove [0,0,0] in markers and replace them with NaN
                points_ml_NaN = transforms_zero_to_nan(points_ml)
                #points_ml_filtered = filter_point_data_with_NaN(points_ml_NaN, fq_ml, cutoff=6.0, order=4)
                points_ml_filtered = points_ml_NaN

                # extract from the markerbased trial the cut of the markerless trial
                to_trim, start_idx, end_idx = extract_trim_from_marker_based(mb_trial_path)
                if to_trim is None:
                    print(f"❌ No 'begin' and 'end' events found for {trial_name}. Skipping trimming.")
                    points_ml_trimmed = points_ml_filtered
                else:
                    points_ml_trimmed = points_ml_filtered[:, :, start_idx:end_idx]

                name_points = c3d_ml['parameters']['POINT']['LABELS']['value']

                write_new_c3d(points_ml_trimmed, name_points, fq_ml, str(output_subject_folder / f"{trial_name}.c3d"))


                