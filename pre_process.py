import ezc3d    
from pathlib import Path
# from functions_batch_mb import (
#     filter_point_data,
#     prepro_export_mb,
#     resample_point_data_mb,
#     trim_c3d_trial_mb,
# )
import numpy as np
from utils import (transforms_zero_to_nan, 
                   filter_point_data_with_NaN,
                   extract_trim_from_marker_based,
                   resample_point_data,
                   load_c3d,
                   write_new_c3d,
                   filter_point_data_with_nan_segments)

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
        subject_name_ml = "Sujet_0"+subject_name[-2:]
        ml_subject_folder = folder_markerless_to_correct / subject_name_ml
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue
        
        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_marker_less_to_export / subject_name 
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
    c3d_mb, points_mb, fq_mb, labels_mb = load_c3d(trial)
    c3d_ml,points_ml, fq_ml, labels_ml =load_c3d(ml_trial_path) 

    # get the frame count of the two c3d files
    nb_frames_mb = points_mb.shape[2]
    nb_frames_ml = points_ml.shape[2]
    
    #calculate the ratio of the two frame counts
    ratio = nb_frames_mb / nb_frames_ml
    # find if the ratio is close to 1 or 2  
    ratio_rounded = round(ratio)

    if not np.isclose(ratio, ratio_rounded, atol=0.1):
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

            # Find all C3D files in this subject's 'Final' folder
            c3d_files = list(raw_folder.glob("*.c3d"))
            total_files = len(c3d_files)
            print(f"Found {total_files} C3D files for {subject_name}\n")

            # =========================================================================
            #      D. TRIMMING, NaN VERIFICATION & UPPER LIMBS JOINT DETERMINATION  
            # =========================================================================
            for file_path in c3d_files:
                pre_processed_marker_based_files(file_path =file_path, output_folder = preprocess_output_folder)
                

def pre_processed_marker_based_files(file_path: Path, output_folder: Path): 
    file_name = file_path.name
    target_fps = 60.0  # Target frame rate for resampling
    print(f"Processing: {file_name}")

    c3d, points, fq, labels = load_c3d(file_path)

    # 2. Trim and log
    to_trim, start_idx, end_idx = extract_trim_from_marker_based(file_path)
    if to_trim:
        trim_points = points[:, :, start_idx:end_idx]
    else:
        trim_points = points

    # filter and resample
    filtered_points = filter_point_data_with_nan_segments(points=trim_points,
                                        fs = fq,
                                        cutoff=6.0,
                                        order=2,
                                        max_gap=10)
    resampled_points = resample_point_data(filtered_points,
                                                    original_frame_rate=fq, 
                                                        target_frame_rate=target_fps)

    # TODO : Rajout export points (necessaire ou pas)

    exported_c3d_path =  output_folder / f"{file_path.stem}.c3d"
    name_points = c3d['parameters']['POINT']['LABELS']['value']
    
    write_new_c3d(resampled_points, name_points, target_fps, str(output_folder / f"{file_name}.c3d")) 


def pre_processed_marker_less_folder(folder_marker_based: Path, folder_markerless_to_correct: Path, folder_marker_less_to_export: Path,
                                    remove_nan: bool = True,
                                    filter_data: bool = True,
                                    cutoff: float = 6.0,
                                    order: int = 4):
    """
    """
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
                pre_processed_marker_less_files(ml_trial_path, mb_trial_path,output_subject_folder,
                                                remove_nan=remove_nan,
                                                filter_data=filter_data,
                                                cutoff=cutoff,
                                                order=order)


def pre_processed_marker_less_files(ml_trial_path: Path, mb_trial_path: Path, output_subject_folder: Path,
                                    remove_nan: bool = True,
                                    filter_data: bool = True,
                                    cutoff: float = 6.0,
                                    order: int = 4):
    """
    Pre-processing of a single file of markerless data based on the corresponding marker-based data.
    """
    trial_name = ml_trial_path.stem
    print(f"Processing trial: {trial_name}")
    # Check if the corresponding markerless trial exists
    
    # remove NaN from the markerless trial
    c3d_ml, points_ml, fq_ml, name_points = load_c3d(ml_trial_path)

    # Remove [0,0,0] in markers and replace them with NaN
    if remove_nan:
        points_ml_NaN = transforms_zero_to_nan(points_ml)
    else:
        points_ml_NaN = points_ml

    if filter_data:
        points_ml_filtered = filter_point_data_with_nan_segments(points_ml_NaN,
                                                                fq_ml, 
                                                                cutoff=cutoff, 
                                                                order=order,
                                                                max_gap=10)
    else:
        points_ml_filtered = points_ml_NaN

    # extract from the markerbased trial the cut of the markerless trial
    to_trim, start_idx, end_idx = extract_trim_from_marker_based(mb_trial_path)
    # TODO : CHECK if the trim is at the same frame rate.
    if to_trim is None:
        print(f"❌ No 'begin' and 'end' events found for {trial_name}. Skipping trimming.")
        points_ml_trimmed = points_ml_filtered
    else:
        points_ml_trimmed = points_ml_filtered[:, :, start_idx:end_idx]

    write_new_c3d(points_ml_trimmed, name_points, fq_ml, str(output_subject_folder / f"{trial_name}.c3d"))
