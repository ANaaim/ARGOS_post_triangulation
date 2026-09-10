import ezc3d    
from pathlib import Path

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


def trim_marker_less_data_folder(folder_marker_less: Path, folder_marker_based: Path, folder_export: Path):
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
        ml_subject_folder = folder_marker_less / subject_name 
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue
        
        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_export / subject_name 
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
            trim_marker_less_data_file(trial, ml_trial_path, output_subject_folder)

def trim_marker_less_data_file(trial_marker_less: Path, trial_marker_based: Path, output_folder: Path):
    trial_name = trial_marker_less.stem
    processing_info = f"Processing trial: {trial_name}"
    print(processing_info)
    # TODO: Implement the trimming logic for the marker-less data
    pass