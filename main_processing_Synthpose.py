from create_model_upperlimb import model_creation_from_measured_data
from pathlib import Path
import kinematics
import create_model_upperlimb as marker_based
import create_model_upperlimb_SynthRTM as markerless

import os
from pathlib import Path

import kinematics

# ---------------------------------------------------------------------
# Root data folder
root_folder_marker_less = Path(r".\data\pre_processed\marker_less\SynthRTMPose")
root_folder_marker_based = Path(r".\data\pre_processed\marker_based")
#root_folder_fake_marker_less = Path(r".\Data\test\marker_less_from_marker_based")
# You may adjust if static files have a specific naming pattern
STATIC_KEYWORD = "00-static-stand.c3d"  # example: "static.c3d"
# Or if your static file is always named 'task.c3d', replace with: STATIC_KEYWORD = "task.c3d"
# ---------------------------------------------------------------------

show_animation = False   # Set to True if you want to visualize the model creation and kinematics results
two_dof_hand_model = True  # Set to True if you want to model the hand with 2 DOF (flexion/extension + deviation), False for 1 DOF (flexion/extension only)
process_marker_based = True  # Set to True if you want to process marker-based data
process_marker_less = False  # Set to True if you want to process marker-less data

# TODO : mieux définir les noms des dossier de sortie.

if two_dof_hand_model:
    folder_to_save_ml = Path(r".\Kinematics\Synthpose\with_2dof_hand")
    folder_to_save_mb = Path(r".\Kinematics\Marker_based\with_2dof_hand")
else:
    folder_to_save_ml = Path(r".\Kinematics\Synthpose\with_1dof_hand")
    folder_to_save_mb = Path(r".\Kinematics\Marker_based\with_1dof_hand")
# check if folder exist 
if not folder_to_save_ml.exists():
    folder_to_save_ml.mkdir(parents=True, exist_ok=True)
if not folder_to_save_mb.exists():
    folder_to_save_mb.mkdir(parents=True, exist_ok=True)


if process_marker_less:
    for subject_folder in root_folder_marker_less.iterdir():
        if not subject_folder.is_dir():
            continue

        print(f"\n--- Processing subject: {subject_folder.name} ---")

        # ------------------------------------------------------------
        # 1. Locate static trial
        # ------------------------------------------------------------
        static_trial = None
        for file in subject_folder.glob("*.c3d"):
            if STATIC_KEYWORD in file.name.lower():
                static_trial = file
                break

        if static_trial is None:
            print(f"❌ No static trial found for {subject_folder.name}")
            continue

        print(f"✔ Static trial found: {static_trial.name}")

        # ------------------------------------------------------------
        # 2. Create model
        # ------------------------------------------------------------
        # Left
        model_name_L = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_L"
        model_name_R = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_R"
        model_name_RL = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_RL"
        # check if the folder of the output file exist, if not create it
        if not model_name_L.parent.exists():
            model_name_L.parent.mkdir(parents=True, exist_ok=True)
        print("⚙ Creating model...")
        markerless.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_L),
            side_to_process="left",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_L.name}")
        # Right
        if not model_name_R.parent.exists():
            model_name_R.parent.mkdir(parents=True, exist_ok=True)
        markerless.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_R),
            side_to_process="right",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_R.name}")

        # Both
        if not model_name_RL.parent.exists():
            model_name_RL.parent.mkdir(parents=True, exist_ok=True)

        markerless.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_RL),
            side_to_process="right_and_left",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_RL.name}")
        # ------------------------------------------------------------
        # 3. Process all task trials
        # ------------------------------------------------------------
        model_name_L = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_L.bioMod"
        model_name_R = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_R.bioMod"
        model_name_RL = folder_to_save_ml / "markerless" / subject_folder.name / "markerless_model_RL.bioMod"
        for task_file in subject_folder.glob("*.c3d"):
            if STATIC_KEYWORD in task_file.name.lower():
                continue  # skip static trial

            print(f"⚙ Processing task: {task_file.name}")
            file_exported_L = folder_to_save_ml / "markerless" / subject_folder.name / f"{task_file.stem}_L.csv"
            # check if the folder of the output file exist, if not create it
            if not file_exported_L.parent.exists():
                file_exported_L.parent.mkdir(parents=True, exist_ok=True)
            kinematics.main(
                str(task_file),
                str(model_name_L),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_L),
            )
            file_exported_R = folder_to_save_ml / "markerless" / subject_folder.name / f"{task_file.stem}_R.csv"
            # check if the folder of the output file exist, if not create it
            if not file_exported_R.parent.exists():
                file_exported_R.parent.mkdir(parents=True, exist_ok=True)
            kinematics.main(
                str(task_file),
                str(model_name_R),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_R),
            )
            file_exported_RL = folder_to_save_ml / "markerless" / subject_folder.name / f"{task_file.stem}_RL.csv"
            # check if the folder of the output file exist, if not create it 
            if not file_exported_RL.parent.exists():
                file_exported_RL.parent.mkdir(parents=True, exist_ok=True)
            kinematics.main(
                str(task_file),
                str(model_name_RL),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_RL),
            )
            print(f"✔ Finished: {task_file.name}")



if process_marker_based:
    for subject_folder in root_folder_marker_based.iterdir():
        if not subject_folder.is_dir():
            continue

        print(f"\n--- Processing subject: {subject_folder.name} ---")

        # ------------------------------------------------------------
        # 1. Locate static trial
        # ------------------------------------------------------------
        static_trial = None
        for file in subject_folder.glob("*.c3d"):
            if STATIC_KEYWORD in file.name.lower():
                static_trial = file
                break

        if static_trial is None:
            print(f"❌ No static trial found for {subject_folder.name}")
            continue

        print(f"✔ Static trial found: {static_trial.name}")

        # ------------------------------------------------------------
        # 2. Create model
        # ------------------------------------------------------------
        model_name_R = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_R"
        model_name_L = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_L"
        model_name_RL = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_RL"
        # check if the folder of the output file exist, if not create it
        if not model_name_R.parent.exists():
            model_name_R.parent.mkdir(parents=True, exist_ok=True)
        if not model_name_L.parent.exists():
            model_name_L.parent.mkdir(parents=True, exist_ok=True)
        if not model_name_RL.parent.exists():
            model_name_RL.parent.mkdir(parents=True, exist_ok=True)

        print("⚙ Creating model...")
        marker_based.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_R),
            side_to_process="right",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_R.name}")
        marker_based.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_L),
            side_to_process="left",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_L.name}")
        marker_based.model_creation_from_measured_data(
            static_trial_path=str(static_trial),
            model_name=str(model_name_RL),
            side_to_process="right_left",
            two_dof_hand_model=two_dof_hand_model,
            animate_model=show_animation,  # You can set True if you want visualization
        )
        print(f"✔ Model created: {model_name_RL.name}")

        # ------------------------------------------------------------
        # 3. Process all task trials
        # ------------------------------------------------------------
        model_name_R = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_R.bioMod"
        model_name_L = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_L.bioMod"
        model_name_RL = folder_to_save_mb / "marker_based" / subject_folder.name / "markerbased_model_RL.bioMod"
        # check if the folder of the output file exist, if not create it
        if not model_name_R.parent.exists():
            model_name_R.parent.mkdir(parents=True, exist_ok=True)
        if not model_name_L.parent.exists():
            model_name_L.parent.mkdir(parents=True, exist_ok=True)
        if not model_name_RL.parent.exists():
            model_name_RL.parent.mkdir(parents=True, exist_ok=True)
        for task_file in subject_folder.glob("*.c3d"):
            # if STATIC_KEYWORD in task_file.name.lower():
            #    continue  # skip static trial

            print(f"⚙ Processing task: {task_file.name}")
            file_exported_R = folder_to_save_mb / "marker_based" / subject_folder.name / f"{task_file.stem}_R.csv"
            file_exported_L = folder_to_save_mb / "marker_based" / subject_folder.name / f"{task_file.stem}_L.csv"
            file_exported_RL = folder_to_save_mb / "marker_based" / subject_folder.name / f"{task_file.stem}_RL.csv"
            # check if the folder of the output file exist, if not create it
            if not file_exported_R.parent.exists():
                file_exported_R.parent.mkdir(parents=True, exist_ok=True)
            if not file_exported_L.parent.exists():
                file_exported_L.parent.mkdir(parents=True, exist_ok=True)
            if not file_exported_RL.parent.exists():
                file_exported_RL.parent.mkdir(parents=True, exist_ok=True)

            kinematics.main(
                str(task_file),
                str(model_name_R),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_R),
            )
            kinematics.main(
                str(task_file),
                str(model_name_L),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_L),
            )   
            kinematics.main(
                    str(task_file),
                    str(model_name_RL),
                    show=show_animation,  # Set True if you want to visualize the kinematics results
                    filename_output=str(file_exported_RL),
                )
            print(f"✔ Finished: {task_file.name}")


print("\n=== All subjects processed successfully ===")
