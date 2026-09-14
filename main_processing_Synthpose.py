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

if two_dof_hand_model:
    folder_to_save = Path(r".\Kinematics\Synthpose\with_2dof_hand")
else:
    folder_to_save = Path(r".\Kinematics\Synthpose\with_1dof_hand")
# check if folder exist 
if not folder_to_save.exists():
    folder_to_save.mkdir(parents=True, exist_ok=True)

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
    model_name = folder_to_save / "markerless" / subject_folder.name / "markerless_model"
    # check if the folder of the output file exist, if not create it
    if not model_name.parent.exists():
        model_name.parent.mkdir(parents=True, exist_ok=True)
    print("⚙ Creating model...")
    markerless.model_creation_from_measured_data(
        static_trial_path=str(static_trial),
        model_name=str(model_name),
        two_dof_hand_model=two_dof_hand_model,
        animate_model=show_animation,  # You can set True if you want visualization
    )
    print(f"✔ Model created: {model_name.name}")

    # ------------------------------------------------------------
    # 3. Process all task trials
    # ------------------------------------------------------------
    model_name = folder_to_save / "markerless" / subject_folder.name / "markerless_model.bioMod"
    for task_file in subject_folder.glob("*.c3d"):
        if STATIC_KEYWORD in task_file.name.lower():
            continue  # skip static trial

        print(f"⚙ Processing task: {task_file.name}")
        file_exported = folder_to_save / "markerless" / subject_folder.name / f"{task_file.stem}.csv"
        # check if the folder of the output file exist, if not create it
        if not file_exported.parent.exists():
            file_exported.parent.mkdir(parents=True, exist_ok=True)
        kinematics.main(
            str(task_file),
            str(model_name),
            show=show_animation,  # Set True if you want to visualize the kinematics results
            filename_output=str(file_exported),
        )

        print(f"✔ Finished: {task_file.name}")

# for subject_folder in root_folder_marker_based.iterdir():
#     if not subject_folder.is_dir():
#         continue

#     print(f"\n--- Processing subject: {subject_folder.name} ---")

#     # ------------------------------------------------------------
#     # 1. Locate static trial
#     # ------------------------------------------------------------
#     static_trial = None
#     for file in subject_folder.glob("*.c3d"):
#         if STATIC_KEYWORD in file.name.lower():
#             static_trial = file
#             break

#     if static_trial is None:
#         print(f"❌ No static trial found for {subject_folder.name}")
#         continue

#     print(f"✔ Static trial found: {static_trial.name}")

#     # ------------------------------------------------------------
#     # 2. Create model
#     # ------------------------------------------------------------
#     model_name = folder_to_save / "marker_based" / subject_folder.name / "markerbased_model"
#     # check if the folder of the output file exist, if not create it
#     if not model_name.parent.exists():
#         model_name.parent.mkdir(parents=True, exist_ok=True)
#     print("⚙ Creating model...")
#     marker_based.model_creation_from_measured_data(
#         static_trial_path=str(static_trial),
#         model_name=str(model_name),
#         two_dof_hand_model=two_dof_hand_model,
#         animate_model=show_animation,  # You can set True if you want visualization
#     )
#     print(f"✔ Model created: {model_name.name}")

#     # ------------------------------------------------------------
#     # 3. Process all task trials
#     # ------------------------------------------------------------
#     model_name = folder_to_save / "marker_based" / subject_folder.name / "markerbased_model.bioMod"
#     # check if the folder of the output file exist, if not create it
#     if not model_name.parent.exists():
#         model_name.parent.mkdir(parents=True, exist_ok=True)
#     for task_file in subject_folder.glob("*.c3d"):
#         # if STATIC_KEYWORD in task_file.name.lower():
#         #    continue  # skip static trial

#         print(f"⚙ Processing task: {task_file.name}")
#         file_exported = folder_to_save / "marker_based" / subject_folder.name / f"{task_file.stem}.csv"
#         # check if the folder of the output file exist, if not create it
#         if not file_exported.parent.exists():
#             file_exported.parent.mkdir(parents=True, exist_ok=True)
#         kinematics.main(
#             str(task_file),
#             str(model_name),
#             show=show_animation,  # Set True if you want to visualize the kinematics results
#             filename_output=str(file_exported),
#         )

#         print(f"✔ Finished: {task_file.name}")


print("\n=== All subjects processed successfully ===")
