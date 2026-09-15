"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os

import numpy as np
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
    RotoTransMatrix,
)
from pathlib import Path


def model_creation_from_measured_data(
    static_trial_path: Path, model_name: str, side_to_process: str, two_dof_hand_model: bool = False, animate_model: bool = True
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
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "T10", "IJ", "PX"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers(["PX", "T10"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["C7", "IJ"]),
                ),
                second_axis=Axis(name=Axis.Name.X, start="C7", end="IJ"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("IJ", "PX", "C7", "T10"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("IJ", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("PX", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))

    if "right" in side_to_process.lower():
        reduced_model.add_segment(
            Segment(
                name="R_Humerus_1",
                parent_name="Thorax",
                rotations=Rotations.X,
                dof_names=["R_Shoulder_AddAbd"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
            )
        )
        reduced_model.add_segment(
            Segment(
                name="R_Humerus_2",
                parent_name="R_Humerus_1",
                rotations=Rotations.Z,
                dof_names=["R_Shoulder_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
            )
        )
        reduced_model.add_segment(
            Segment(
                name="R_Humerus_3",
                parent_name="R_Humerus_2",
                rotations=Rotations.Y,
                dof_names=["R_Shoulder_Rot"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
            )
        )

        reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_EM", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_EL", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_Up_tech_arm", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_Med_tech_arm", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_1",
                parent_name="R_Humerus_3",
                rotations=Rotations.Z,
                dof_names=["R_Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("R_EL", "R_EL"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_1"].add_marker(Marker("R_EL", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Forearm_1"].add_marker(Marker("R_EM", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["R_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_US", end="R_RS"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_US", "R_RS"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_US", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_RS", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["R_Wrist_FleExt", "R_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["R_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="R_Hand",
                parent_name="R_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_HM2", "R_HM5"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_HM5", end="R_HM2"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_US", "R_RS"), is_local=False),
            )
        )
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM2", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM5", is_technical=True, is_anatomical=True))

    if "left" in side_to_process.lower():
        reduced_model.add_segment(
            Segment(
                name="L_Humerus_1",
                parent_name="Thorax",
                rotations=Rotations.X,
                dof_names=["L_Shoulder_AddAbd"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
            )
        )
        reduced_model.add_segment(
            Segment(
                name="L_Humerus_2",
                parent_name="L_Humerus_1",
                rotations=Rotations.Z,
                dof_names=["L_Shoulder_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
            )
        )
        reduced_model.add_segment(
            Segment(
                name="L_Humerus_3",
                parent_name="L_Humerus_2",
                rotations=Rotations.Y,
                dof_names=["L_Shoulder_Rot"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_GH",
                    first_axis=Axis(
                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
            )
        )

        reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_EM", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_EL", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_tech_up_arm", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_tech_med_arm", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_1",
                parent_name="L_Humerus_3",
                rotations=Rotations.Z,
                dof_names=["L_Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("L_EL", "L_EM"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_1"].add_marker(Marker("L_EL", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Forearm_1"].add_marker(Marker("L_EM", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_RS", end="L_US"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_US", "L_RS"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_US", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_RS", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["L_Wrist_FleExt", "L_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["L_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="L_Hand",
                parent_name="L_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_HM2", "L_HM5"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_HM2", end="L_HM5"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_US", "L_RS"), is_local=False),
            )
        )
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM2", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM5", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real
