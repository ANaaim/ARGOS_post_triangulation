from pathlib import Path
import create_model_upperlimb
import kinematics


def mbo(
    root_folder_marker_less: Path,
    static_filename: str,
    model_creation,
    folder_to_save: Path,
    two_dof_hand_model: bool = True,
    show_animation: bool = False,
):
    for subject_folder in root_folder_marker_less.iterdir():
        if not subject_folder.is_dir():
            continue

        print(f"\n--- Processing subject: {subject_folder.name} ---")

        # ------------------------------------------------------------
        # 1. Locate static trial
        # ------------------------------------------------------------
        static_trial = None
        for file in subject_folder.glob("*.c3d"):
            if static_filename in file.name.lower():
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
        model_name_L = folder_to_save / subject_folder.name / "model_L"
        model_name_R = folder_to_save / subject_folder.name / "model_R"
        model_name_RL = folder_to_save / subject_folder.name / "model_RL"
        # check if the folder of the output file exist, if not create it
        if not model_name_L.parent.exists():
            model_name_L.parent.mkdir(parents=True, exist_ok=True)
        print("⚙ Creating model...")
        model_creation(
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
        model_creation(
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

        model_creation(
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
        model_name_L = folder_to_save / subject_folder.name / "model_L.bioMod"
        model_name_R = folder_to_save / subject_folder.name / "model_R.bioMod"
        model_name_RL = folder_to_save / subject_folder.name / "model_RL.bioMod"

        for task_file in subject_folder.glob("*.c3d"):
            if static_filename in task_file.name.lower():
                continue  # skip static trial

            print(f"⚙ Processing task: {task_file.name}")
            file_exported_L = folder_to_save / subject_folder.name / f"{task_file.stem}_L.csv"
            # check if the folder of the output file exist, if not create it
            if not file_exported_L.parent.exists():
                file_exported_L.parent.mkdir(parents=True, exist_ok=True)
            kinematics.main(
                str(task_file),
                str(model_name_L),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_L),
            )
            file_exported_R = folder_to_save / subject_folder.name / f"{task_file.stem}_R.csv"
            # check if the folder of the output file exist, if not create it
            if not file_exported_R.parent.exists():
                file_exported_R.parent.mkdir(parents=True, exist_ok=True)
            kinematics.main(
                str(task_file),
                str(model_name_R),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str(file_exported_R),
            )
            file_exported_RL = folder_to_save / subject_folder.name / f"{task_file.stem}_RL.csv"
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


if __name__ == "__main__":
    # Example usage

    base_save_folder = Path(r".\Kinematics")
    two_dof_hand_model = True  # Set to True if you want to model the hand with 2 DOF (flexion/extension + deviation), False for 1 DOF (flexion/extension only)
    static_filename = "00-static-stand.c3d"  # Adjust if your static file has a different name

    # Marker based
    root_folder_marker_based = Path(r".\data\pre_processed\marker_based")
    folder_to_save_marker_based = (
        base_save_folder / "Marker_based" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "Marker_based" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.marker_based # Use the appropriate function for model creation
    mbo(
        root_folder_marker_based,
        static_filename,
        model_creation,
        folder_to_save_marker_based,
        two_dof_hand_model=two_dof_hand_model,
        show_animation=False,
    )

    # SynthRTMPose
    root_folder_SynthRTMPose = Path(r".\data\pre_processed\marker_less\SynthRTMPose")
    folder_to_save_SynthRTMPose = (
        base_save_folder / "SynthRTMPose" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "SynthRTMPose" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.synthRTM  # Use the appropriate function for model creation
    mbo(
        root_folder_SynthRTMPose,
        static_filename,
        model_creation,
        folder_to_save_SynthRTMPose,
        two_dof_hand_model=two_dof_hand_model,
        show_animation=False,
    )

    # Synthpose
    root_folder_SynthPose = Path(r".\data\pre_processed\marker_less\SynthPose")
    folder_to_save_SynthPose = (
        base_save_folder / "Synthpose" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "Synthpose" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.synthpose  # Use the appropriate function for model creation
    mbo(
        root_folder_SynthPose,
        static_filename,
        model_creation,
        folder_to_save_SynthPose,
        two_dof_hand_model=two_dof_hand_model,
        show_animation=False,
    )
