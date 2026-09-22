from pathlib import Path

import numpy as np
import ezc3d
import snip_ezc3d as snip

path_data = Path(".\\data\\pre_processed")
list_model_ML = ["SynthPose", "RTMPose"]




dict_anatomical_frame = {"R_GH":"thorax",
                         "L_GH":"thorax",
                         "R_EJC":"humerus",
                         "L_EJC":"humerus",
                         "R_WJC":"forearm",
                         "L_WJC":"forearm",
                         "R_FJC":"hand",
                         "L_FJC":"hand"}


# For each point the error will be calculated in the absolute X,Y,Z coordinates and in the segment frame 
# R_GH et L_GH in the thorax frame 
# R_EJC et L_EJC in the humerus frame
# R_WJC et L_WJC in the forearm frame
# In addition for synthpose the error of all the point with and equivalent in the marker based model will be calculated 
# in a similar fashion
# C7 ==> C7 
# T11 ==> T10 (attention placement plus difficile)
# sternum ==> IJ
# ==> X_EL
# ==> X_EM
# ==> R_RS
# ==> R_US
# For RTM pose the position of the equivalent of HM2 and HM5 will be calculated in 
# the hand frame and the absolute frame 

# for each model we will stock a dictionary of the comparison results ==> let'use polars to make comparison faster 
# for all file in marker based data we try to find the corresponding file in the marker less data and compare the points

    

def compare_absolute_file(path_marker_based : Path, path_synth: Path, path_RTM: Path, dict_Synth : dict, dict_RTM : dict):
    # compare the absolute position of the points in the marker based data and the marker less data
    acq_Synth = ezc3d.c3d(str(path_synth))
    acq_RTM = ezc3d.c3d(str(path_RTM))
    acq_MB = ezc3d.c3d(str(path_marker_based))


    points_data_Synth, points_name_Synth, points_Synth = snip.get_points_ezc3d(acq_Synth)
    points_data_RTM, points_name_RTM, points_RTM = snip.get_points_ezc3d(acq_RTM)
    points_data_MB, points_name_MB, points_MB = snip.get_points_ezc3d(acq_MB)
    dict_error_Synth = {}
    dict_error_RTM = {}
    for point_MB, point_synth in dict_Synth.items():
        if point_synth in points_name_Synth and point_MB in points_name_MB:
            index_point_synth = points_name_Synth.index(point_synth)
            index_point_MB = points_name_MB.index(point_MB)
            point_data_synth = points_data_Synth[:, index_point_synth, :]
            point_data_MB = points_data_MB[:, index_point_MB, :]
            error = point_data_synth - point_data_MB
            dict_error_Synth[point_MB] = error

    for point_MB, point_RTM in dict_RTM.items():
        if point_RTM in points_name_RTM and point_MB in points_name_MB:
            index_point_RTM = points_name_RTM.index(point_RTM)
            index_point_MB = points_name_MB.index(point_MB)
            point_data_RTM = points_data_RTM[:, index_point_RTM, :]
            point_data_MB = points_data_MB[:, index_point_MB, :]
            error = point_data_RTM - point_data_MB
            dict_error_RTM[point_MB] = error


    return dict_error_Synth, dict_error_RTM

def compare_anatomical_frame_file(path_synth: Path, path_RTM: Path, path_marker_based : Path):
    # compare the position of the points in the marker based data and the marker less data in the anatomical fram
    pass


import polars as pl


def error_dict_to_dataframe(
    error_dict: dict,
    subject: str,
    task: str,
    coordinate_system: str,
    model: str,
) -> pl.DataFrame:

    dfs = []

    for point, error in error_dict.items():

        # error est supposé être de forme (3, n_frames)
        n_frames = error.shape[1]

        df = pl.DataFrame({
            "subject": [subject] * n_frames,
            "task": [task] * n_frames,
            "coordinate_system": [coordinate_system] * n_frames,
            "model": [model] * n_frames,
            "point": [point] * n_frames,
            "frame": np.arange(n_frames),
            "error_x": error[0, :],
            "error_y": error[1, :],
            "error_z": error[2, :],
        }).with_columns(
            (
                pl.col("error_x")**2
                + pl.col("error_y")**2
                + pl.col("error_z")**2
            ).sqrt().alias("error_norm")
        )

        dfs.append(df)

    if not dfs:
       return pl.DataFrame()

    return pl.concat(dfs)


if __name__ == "__main__":
    path_RTM = path_data / "marker_less" / "RTMPose"
    path_synth = path_data / "marker_less" / "SynthPose"
    path_marker_based = path_data / "marker_based"


    # a dictionary allowing to define which point is compared with each point
    dict_comparaison_point_Synth = {"R_GH":"R_Shoulder",
                                    "L_GH":"L_Shoulder",
                                    "R_EJC":"R_Elbow",
                                    "L_EJC":"L_Elbow",
                                    "R_WJC":"R_Wrist",
                                    "L_WJC":"L_Wrist",
                                    "IJ":"sternum",
                                    "C7":"C7",
                                    "T10":"T11",
                                    "R_EL":"r_lelbow",
                                    "R_EM":"r_melbow",
                                    "L_EL":"l_lelbow",
                                    "L_EM":"l_melbow",
                                    "R_RS":"r_lwrist",
                                    "R_US":"r_mwrist",
                                    "L_RS":"l_lwrist",
                                    "L_US":"l_mwrist",}

    dict_comparaison_point_RTM = {"R_GH":"R_shoulder",
                              "L_GH":"L_shoulder",
                              "R_EJC":"R_elbow",
                              "L_EJC":"L_elbow",
                              "R_WJC":"R_wrist",
                              "L_WJC":"L_wrist",
                              "R_HM2":"R_MCP_index",
                              "L_HM2":"L_MCP_index",
                              "R_HM5":"R_MCP_little",
                              "L_HM5":"L_MCP_little"}

    dict_frame_to_use = {"R_GH":"thorax",
                         "L_GH":"thorax",
                         "R_EJC":"humerus",
                         "L_EJC":"humerus",
                         "R_WJC":"forearm",
                         "L_WJC":"forearm",
                         "R_HM2":"hand",
                         "L_HM2":"hand",
                         "R_HM5":"hand",
                         "L_HM5":"hand",
                         "IJ":"thorax",
                         "C7":"thorax",
                         "T10":"thorax",
                         "R_EL":"humerus",
                         "R_EM":"humerus",
                         "L_EL":"humerus",
                         "L_EM":"humerus",
                         "R_RS":"r_lwrist",
                         "R_US":"r_mwrist",
                         "L_RS":"l_lwrist",
                         "L_US":"l_mwrist",}

    for subject in path_marker_based.iterdir():
        for task_file in (subject).glob("*.c3d"):
            if (path_synth / subject.name / task_file.name).exists() and (path_RTM / subject.name / task_file.name).exists():
                print(f"Comparing {task_file.name} for subject {subject.name}")
                dict_error_Synth, dict_error_RTM = compare_absolute_file(path_marker_based / subject.name / task_file.name,
                                                                        path_synth / subject.name / task_file.name,
                                                                        path_RTM / subject.name / task_file.name,
                                                                        dict_comparaison_point_Synth,
                                                                        dict_comparaison_point_RTM)
                # create a parquet file to stock the error for each point in the marker based data and the marker less data
                df_error_Synth = error_dict_to_dataframe(dict_error_Synth, subject.name, task_file.stem,"absolute", "SynthPose")
                df_error_RTM = error_dict_to_dataframe(dict_error_RTM, subject.name, task_file.stem, "absolute", "RTMPose")
                

                all_errors.extend([df_error_Synth, df_error_RTM])

    # Combine toutes les erreurs
    df_errors = pl.concat(all_errors)
    # Sauvegarde
    output_file = path_data / "absolute_position_errors.parquet"
    df_errors.write_parquet(output_file)
    print(f"Errors saved to: {output_file}")
    print(df_errors)
    print(f"There is {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")

