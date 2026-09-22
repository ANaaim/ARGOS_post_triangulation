from pathlib import Path
from matplotlib.axis import Axis
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip


def compare_absolute_file(
        path_marker_based : Path, 
        path_synth: Path, 
        path_RTM: Path, 
        dict_Synth : dict, 
        dict_RTM : dict) -> tuple[dict, dict]:
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

def compare_anatomical_frame_file(
    path_synth: Path,
    path_RTM: Path,
    path_marker_based : Path,
    dict_Synth : dict, 
    dict_RTM : dict, 
    dict_frame_to_use : dict
    ) -> tuple[dict, dict]:

    acq_Synth = ezc3d.c3d(str(path_synth))
    acq_RTM = ezc3d.c3d(str(path_RTM))
    acq_MB = ezc3d.c3d(str(path_marker_based))
    # compare the position of the points in the marker based data and the marker less data in the anatomical fram
    points_data_Synth, points_name_Synth, points_Synth = snip.get_points_ezc3d(acq_Synth)
    points_data_RTM, points_name_RTM, points_RTM = snip.get_points_ezc3d(acq_RTM)
    points_data_MB, points_name_MB, points_MB = snip.get_points_ezc3d(acq_MB)
    dict_error_Synth = {}
    dict_error_RTM = {}
    for point_MB, point_synth in dict_Synth.items():
        if point_synth in points_name_Synth and point_MB in points_name_MB:
            print(f"Comparing {point_MB} in SynthPose and Marker Based")
            frame_to_use = dict_frame_to_use[point_MB]
            print(f"Using {frame_to_use} anatomical frame for {point_MB}")
            side = point_MB[0]
            
            error = calculate_error_anatomical_frame(points_data_ref = points_data_MB, 
                                                     points_name_ref = points_name_MB, 
                                                     point_name_ref = point_MB,
                                                     points_data = points_data_Synth,
                                                     points_name = points_name_Synth,
                                                     point_name = point_synth,
                                                     frame_to_use=frame_to_use,
                                                     side=side)
            dict_error_Synth[point_MB] = error

    for point_MB, point_RTM in dict_RTM.items():
        if point_RTM in points_name_RTM and point_MB in points_name_MB:
            frame_to_use = dict_frame_to_use[point_MB]
            print(f"Using {frame_to_use} anatomical frame for {point_MB}")
            side = point_MB[0]
            
            # Project the point onto the thorax anatomical frame
            error = calculate_error_anatomical_frame(points_data_ref = points_data_MB, 
                                                     points_name_ref = points_name_MB, 
                                                     point_name_ref = point_MB,
                                                     points_data = points_data_RTM,
                                                     points_name = points_name_RTM,
                                                     point_name = point_RTM,
                                                     frame_to_use=frame_to_use,
                                                     side=side)
            dict_error_Synth[point_MB] = error
            dict_error_RTM[point_MB] = error

    return dict_error_Synth, dict_error_RTM

def thorax_anatomical_frame(points_data, points_name):
    # get the index of the points in the points_name list
    index_IJ = points_name.index("IJ")
    index_C7 = points_name.index("C7")
    index_T10 = points_name.index("T10")
    index_PX = points_name.index("PX")

    # get the position of the points in the points_data array
    IJ = points_data[:, index_IJ, :]
    C7 = points_data[:, index_C7, :]
    T10 = points_data[:, index_T10, :]
    PX = points_data[:, index_PX, :]
    
    # calculate the thorax anatomical frame
    Y_thorax = ((IJ+C7)/2)-(PX+T10)/2
    Y_thorax /= np.linalg.norm(Y_thorax, axis=0)
    Z_thorax = np.cross((C7 - IJ), Y_thorax, axis=0)
    Z_thorax /= np.linalg.norm(Z_thorax, axis=0)
    X_thorax = np.cross(Y_thorax, Z_thorax, axis=0)

    return X_thorax, Y_thorax, Z_thorax

def humerus_anatomical_frame(points_data, points_name, side="R"):
    # get the index of the points in the points_name list
    index_GH = points_name.index(f"{side}_GH")
    index_EL = points_name.index(f"{side}_EL")
    index_EM = points_name.index(f"{side}_EM")

    # get the position of the points in the points_data array
    GH = points_data[:, index_GH, :]
    EL = points_data[:, index_EL, :]
    EM = points_data[:, index_EM, :]

    # calculate the humerus anatomical frame
    Y_humerus = (GH-(EL+EM)/2)
    Y_humerus /= np.linalg.norm(Y_humerus, axis=0)
    X_humerus = np.cross(Y_humerus,(EL - EM), axis=0)
    X_humerus /= np.linalg.norm(X_humerus, axis=0)
    Z_humerus = np.cross(X_humerus, Y_humerus, axis=0)

    if side == "L":
        X_humerus = -X_humerus
        Z_humerus = -Z_humerus 

    return X_humerus, Y_humerus, Z_humerus

def forearm_anatomical_frame(points_data, points_name, side="R"):
    # get the index of the points in the points_name list
    index_EL = points_name.index(f"{side}_EL")
    index_EM = points_name.index(f"{side}_EM")
    index_RS = points_name.index(f"{side}_RS")
    index_US = points_name.index(f"{side}_US")

    # get the position of the points in the points_data array
    EL = points_data[:, index_EL, :]
    EM = points_data[:, index_EM, :]
    RS = points_data[:, index_RS, :]
    US = points_data[:, index_US, :]

    # calculate the forearm anatomical frame
    Y_forearm = (RS+US)/2-(EL+EM)/2
    Y_forearm /= np.linalg.norm(Y_forearm, axis=0)
    X_forearm = np.cross(Y_forearm, (RS - US), axis=0)
    X_forearm /= np.linalg.norm(X_forearm, axis=0)
    Z_forearm = np.cross(X_forearm, Y_forearm, axis=0)

    if side == "L":
        X_forearm = -X_forearm
        Z_forearm = -Z_forearm 

    return X_forearm, Y_forearm, Z_forearm

def hand_anatomical_frame(points_data, points_name, side="R"):
    # get the index of the points in the points_name list
    index_HM2 = points_name.index(f"{side}_HM2")
    index_HM5 = points_name.index(f"{side}_HM5")
    index_RS = points_name.index(f"{side}_RS")
    index_US = points_name.index(f"{side}_US")

    # get the position of the points in the points_data array
    HM2 = points_data[:, index_HM2, :]
    HM5 = points_data[:, index_HM5, :]
    RS = points_data[:, index_RS, :]
    US = points_data[:, index_US, :]

    # calculate the hand anatomical frame
    Y_hand = (RS+US)/2-(HM2+HM5)/2
    Y_hand /= np.linalg.norm(Y_hand, axis=0)
    X_hand = np.cross(Y_hand, (HM2 - HM5), axis=0)
    X_hand /= np.linalg.norm(X_hand, axis=0)
    Z_hand = np.cross(X_hand, Y_hand, axis=0)

    if side == "L":
        X_hand = -X_hand
        Z_hand = -Z_hand 

    return X_hand, Y_hand, Z_hand

def extract_anatomical_frame(points_data, points_name, frame_to_use, side="R"):
      # Assuming the first character indicates the side (R or L)
    if frame_to_use == "thorax":
        X_MB, Y_MB, Z_MB = thorax_anatomical_frame(points_data, points_name)
    if frame_to_use == "humerus":
        X_MB, Y_MB, Z_MB = humerus_anatomical_frame(points_data, points_name, side=side)
    if frame_to_use == "forearm":
        X_MB, Y_MB, Z_MB = forearm_anatomical_frame(points_data, points_name, side=side)
    if frame_to_use == "hand":
        X_MB, Y_MB, Z_MB = hand_anatomical_frame(points_data, points_name, side=side)
    return X_MB, Y_MB, Z_MB

def project_point_onto_frame(point_data, X_frame, Y_frame, Z_frame):
    point_data_proj = np.array([np.einsum("ij,ij->j", point_data, X_frame),
                                np.einsum("ij,ij->j", point_data, Y_frame),
                                np.einsum("ij,ij->j", point_data, Z_frame),])
    return point_data_proj

def calculate_error_anatomical_frame(points_data_ref : np.ndarray, points_name_ref : list, point_name_ref : str,
                                     points_data: np.ndarray, points_name: list, point_name: str,
                                     frame_to_use: str, side="R"):

    X_frame, Y_frame, Z_frame = extract_anatomical_frame(points_data_ref, points_name_ref, frame_to_use, side=side)

    index_point_synth = points_name.index(point_name)
    index_point_MB = points_name_ref.index(point_name_ref)

    point_coordinate = points_data[:, index_point_synth, :]
    point_ref = points_data_ref[:, index_point_MB, :]

    point_proj = project_point_onto_frame(point_coordinate, X_frame, Y_frame, Z_frame)
    point_ref_proj = project_point_onto_frame(point_ref, X_frame, Y_frame, Z_frame)

    error = point_proj - point_ref_proj

    return error


def error_dict_to_dataframe(
    error_dict: dict,
    subject: str,
    task: str,
    coordinate_system: str,
    dict_frame_to_use:dict,
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
            "frame_type": [dict_frame_to_use.get(point, "unknown")] * n_frames,
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

def main():
    path_data = Path(".\\data\\pre_processed")
    list_model_ML = ["SynthPose", "RTMPose"]
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
                         "R_RS":"forearm",
                         "R_US":"forearm",
                         "L_RS":"forearm",
                         "L_US":"forearm",}
    all_errors = []
    for subject in path_marker_based.iterdir():
        for task_file in (subject).glob("*.c3d"):
            if (path_synth / subject.name / task_file.name).exists() and (path_RTM / subject.name / task_file.name).exists():
                print(f"Comparing {task_file.name} for subject {subject.name}")
                dict_error_Synth, dict_error_RTM = compare_absolute_file(path_marker_based / subject.name / task_file.name,
                                                                        path_synth / subject.name / task_file.name,
                                                                        path_RTM / subject.name / task_file.name,
                                                                        dict_comparaison_point_Synth,
                                                                        dict_comparaison_point_RTM)
                dict_error_anat_Synth, dict_anat_error_RTM = compare_anatomical_frame_file(path_synth / subject.name / task_file.name,
                                                                                 path_RTM / subject.name / task_file.name,
                                                                                 path_marker_based / subject.name / task_file.name,
                                                                                 dict_comparaison_point_Synth,
                                                                                 dict_comparaison_point_RTM,
                                                                                 dict_frame_to_use)

                # create a parquet file to stock the error for each point in the marker based data and the marker less data
                df_error_Synth = error_dict_to_dataframe(dict_error_Synth, subject.name, task_file.stem,"absolute",dict_frame_to_use=dict_frame_to_use, model="SynthPose")
                df_error_RTM = error_dict_to_dataframe(dict_error_RTM, subject.name, task_file.stem, "absolute",dict_frame_to_use=dict_frame_to_use, model="RTMPose")
                df_error_anat_Synth = error_dict_to_dataframe(dict_error_anat_Synth, subject.name, task_file.stem, "anatomical",dict_frame_to_use=dict_frame_to_use, model="SynthPose")
                df_error_anat_RTM = error_dict_to_dataframe(dict_anat_error_RTM, subject.name, task_file.stem, "anatomical", dict_frame_to_use=dict_frame_to_use, model="RTMPose")


                all_errors.extend([df_error_Synth, df_error_RTM, df_error_anat_Synth, df_error_anat_RTM])
    # Combine toutes les erreurs
    df_errors = pl.concat(all_errors)
    # Sauvegarde
    output_file = path_data / "absolute_position_errors.parquet"
    df_errors.write_parquet(output_file)
    print(f"Errors saved to: {output_file}")
    print(df_errors)
    print(f"There is {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()
