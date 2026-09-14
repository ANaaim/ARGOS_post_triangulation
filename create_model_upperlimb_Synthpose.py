"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os

import numpy as np
import kinematics
from biobuddy import (
    Axis,
    BiomechanicalModel,
    C3dData,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    DeLevaTable,
    Sex,
    SegmentName,
    ViewAs,
    SegmentCoordinateSystemUtils,
    RangeOfMotion,
    Ranges,
    RotoTransMatrix,
)
from pathlib import Path


def model_creation_from_measured_data(
    static_trial_path: Path, model_name: str, animate_model: bool = True
):

    static_trial = C3dData(str(static_trial_path))
    output_model_filepath = f"{model_name}.bioMod"
    # Generate the personalized kinematic model
    reduced_model = BiomechanicalModel()
    reduced_model.add_segment(Segment(name="Ground"))

    ## define a Ground segment based on the orientation of the thorax () + getting the global positoin of head in this frame
    reduced_model.add_segment(
        Segment(
            name="Thorax",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            dof_names=["Thorax_TX", "Thorax_TY", "Thorax_TZ", "Thorax_RX", "Thorax_RY", "Thorax_RZ"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7","sternum"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start= "T11",
                    end="C7",
                ),
                second_axis=Axis(name=Axis.Name.X, start="T11", end="sternum"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("sternum", "C7", "T11"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("sternum", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T11", is_technical=True, is_anatomical=True))

    # GH = lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["R_AC"])(static_trial.values, None) - np.array(
    #     [0.0, 0.00, 0.01, 0.0]
    # )

    reduced_model.add_segment(
        Segment(
            name="R_Humerus_1",
            parent_name="Thorax",
            rotations=Rotations.X,
            dof_names=["R_Shoulder_AddAbd"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="R_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]), end="R_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
        )
    )
    reduced_model.add_segment(
        Segment(
            name="R_Humerus_2",
            parent_name="R_Humerus_1",
            rotations=Rotations.Z,
            dof_names=["R_Shoulder_FleExt"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="R_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]), end="R_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
        )
    )
    reduced_model.add_segment(
        Segment(
            name="R_Humerus_3",
            parent_name="R_Humerus_2",
            rotations=Rotations.Y,
            dof_names=["R_Shoulder_Rot"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="R_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]), end="R_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
        )
    )

    reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))
    reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="R_Forearm_1",
            parent_name="R_Humerus_3",
            rotations=Rotations.Z,
            dof_names=["R_Elbow_FleExt"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers(["r_lwrist", "r_mwrist"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                ),
                second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("r_lelbow", "r_melbow"), is_local=False),
        )
    )
    reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))
    reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
        Segment(
            name="R_Forearm_2",
            parent_name="R_Forearm_1",
            rotations=Rotations.Y,
            q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
            dof_names=["R_Forearm_ProSup"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers(["r_lwrist", "r_mwrist"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                ),
                second_axis=Axis(name=Axis.Name.Z, start="r_mwrist", end="r_lwrist"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("r_lwrist", "r_mwrist"), is_local=False),
        )
    )
    reduced_model.segments["R_Forearm_2"].add_marker(Marker("r_lwrist", is_technical=True, is_anatomical=True))
    reduced_model.segments["R_Forearm_2"].add_marker(Marker("r_mwrist", is_technical=True, is_anatomical=True))


    
    reduced_model.add_segment(
        Segment(
            name="L_Humerus_1",
            parent_name="Thorax",
            rotations=Rotations.X,
            dof_names=["L_Shoulder_AddAbd"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="L_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]), end="L_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
        )
    )
    reduced_model.add_segment(
        Segment(
            name="L_Humerus_2",
            parent_name="L_Humerus_1",
            rotations=Rotations.Z,
            dof_names=["L_Shoulder_FleExt"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="L_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]), end="L_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
        )
    )
    reduced_model.add_segment(
        Segment(
            name="L_Humerus_3",
            parent_name="L_Humerus_2",
            rotations=Rotations.Y,
            dof_names=["L_Shoulder_Rot"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="L_Shoulder",
                first_axis=Axis(
                    name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]), end="L_Shoulder"
                ),
                second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
        )
    )

    reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
    reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))

    reduced_model.add_segment(
            Segment(
                name="L_Forearm_1",
                parent_name="L_Humerus_3",
                rotations=Rotations.Z,
                dof_names=["L_Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["l_lwrist", "l_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("l_lelbow", "l_melbow"), is_local=False),
            )
        )
    reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
    reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
    
    reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["l_lwrist", "l_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="l_lwrist", end="l_mwrist"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("l_lwrist", "l_mwrist"), is_local=False),
            )
        )
    reduced_model.segments["L_Forearm_2"].add_marker(Marker("l_lwrist", is_technical=True, is_anatomical=True))
    reduced_model.segments["L_Forearm_2"].add_marker(Marker("l_mwrist", is_technical=True, is_anatomical=True))

    # if two_dof_hand_model:
    #     rotation_hand = Rotations.ZX
    #     dof_names_hand = ["Wrist_FleExt", "Wrist_Dev"]
    # else:
    #     rotation_hand = Rotations.Z
    #     dof_names_hand = ["Wrist_FleExt"]

    # reduced_model.add_segment(
    #     Segment(
    #         name="Hand",
    #         parent_name="Forearm_2",
    #         rotations=rotation_hand,
    #         dof_names=dof_names_hand,
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
    #             first_axis=Axis(
    #                 name=Axis.Name.Y,
    #                 start=SegmentCoordinateSystemUtils.mean_markers(["R_HM2", "R_HM5"]),
    #                 end=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
    #             ),
    #             second_axis=Axis(name=Axis.Name.Z, start="R_HM5", end="R_HM2"),
    #             axis_to_keep=Axis.Name.Y,
    #         ),
    #         mesh=Mesh(("R_US", "R_RS"), is_local=False),
    #     )
    # )
    # reduced_model.segments["Hand"].add_marker(Marker("R_HM2", is_technical=True, is_anatomical=True))
    # reduced_model.segments["Hand"].add_marker(Marker("R_HM5", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real

if __name__ == "__main__":
    static_file_path = Path(r".\Data\ABCDE_1920px\huggingpose_hdf5_with_rtmpose\Sujet_008\00-static-stand.c3d")
    dynamic_file_path = Path(r".\Data\ABCDE_1920px\huggingpose_hdf5_with_rtmpose\Sujet_008\03-open-bottle.c3d")
    show_animation = True  # Set to True if you want to visualize the kinematics results
    model_creation_from_measured_data(static_file_path, model_name=r"toto_Synthpose", animate_model=True)
    kinematics.main(
                str(dynamic_file_path),
                str("toto_Synthpose.biomod"),
                show=show_animation,  # Set True if you want to visualize the kinematics results
                filename_output=str("toto_Synthpose.csv"),
            )