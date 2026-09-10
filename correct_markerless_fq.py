import ezc3d    
from pathlib import Path

def correct_markerless_fq(folder_marker_based: Path, folder_markerless_to_correct: Path, folder_marker_less_to_export: Path):
    """
    Correct the frame rate of marker-less data based on the corresponding marker-based data.
    """

    # folder_markerless_to_correct = Path(r".\Data\Synthpose_dev\huggingpose")
    # folder_marker_based = Path(r".\Data\Synthpose_dev\marker_based")

    # folder_marker_less_to_export = Path(r".\Data\Synthpose_dev\ML_corrected")

    for subject_folder in folder_marker_based.iterdir():
        subject_name = subject_folder.name
        print(f"\n--- Processing subject: {subject_name} ---")
        # Check if the subject folder exists in the markerless folder
        ml_subject_folder = folder_markerless_to_correct / subject_name
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue
        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_marker_less_to_export / subject_name
        if not output_subject_folder.exists():
            output_subject_folder.mkdir(parents=True, exist_ok=True)
        for trial in subject_folder.glob("*.c3d"):
            trial_name = trial.stem
            print(f"Processing trial: {trial_name}")
            # Check if the corresponding markerless trial exists
            ml_trial_path = ml_subject_folder / f"{trial_name}.c3d"
            if not ml_trial_path.exists():
                print(f"❌ No corresponding markerless trial for {trial_name}")
                continue
            # Read the marker-based and markerless c3d files
            c3d_mb = ezc3d.c3d(str(trial))
            c3d_ml = ezc3d.c3d(str(ml_trial_path))
            # get the frame count of the two c3d files
            a = 1
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
                continue
            # get fq of mb 
            if ratio_rounded == 2:
                c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
                c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
            elif ratio_rounded == 1:
                c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
                c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
            else:
                print(f"❌ The ratio of the frame counts is not 1 or 2 for {trial_name}. Ratio: {ratio}")

            print(str(output_subject_folder / f"{trial_name}.c3d"))
            c3d_ml.write(str(output_subject_folder / f"{trial_name}.c3d"))   