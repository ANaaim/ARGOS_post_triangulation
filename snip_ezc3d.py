import ezc3d
import numpy as np


# add point in the c3d file
def add_point_in_c3d(filename, new_filename, new_point_name, position):
    c3d = ezc3d.c3d(filename)
    existing_points = c3d["data"]["points"]
    n_coordinates, _, n_frames = existing_points.shape

    position = np.asarray(position, dtype=existing_points.dtype)
    if position.ndim == 1:
        position = position.reshape(3, 1)

    if position.shape == (3, n_frames):
        position = position.reshape(3, 1, n_frames)
    elif position.shape != (3, 1, n_frames):
        raise ValueError(f"position must have shape (3, {n_frames}) or (3, 1, {n_frames}), got {position.shape}")

    new_point = np.zeros((n_coordinates, 1, n_frames), dtype=existing_points.dtype)
    new_point[:3, :, :] = position

    # add the new point to the c3d data
    c3d["parameters"]["POINT"]["USED"]["value"] += 1
    c3d["parameters"]["POINT"]["LABELS"]["value"].append(new_point_name)
    c3d["data"]["points"] = np.concatenate((existing_points, new_point), axis=1)
    # delete meta_points so ezc3d will recreate it automatically to match the new number of points
    if "meta_points" in c3d["data"]:
        del c3d["data"]["meta_points"]
    # save the modified c3d file in the same directory with a new name
    c3d.write(new_filename)


def get_points_ezc3d(acq):
    """Points extraction with a dictionnary allowing to find
    the point position in the numpy array using text without
    using a dictionnary"""

    points_name = acq["parameters"]["POINT"]["LABELS"]["value"]

    points_data = acq["data"]["points"][0:3, :, :]
    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point

    return points_data, points_name, points_ind


def add_point_from_dictionary(acq, point_to_add):
    points, points_name, points_ind = get_points_ezc3d(acq)
    # copy points informations
    new_list = points_name.copy()
    new_array = acq["data"]["points"]
    nb_frame = acq["data"]["points"].shape[2]

    for ind_point, (name_point, value_point) in enumerate(point_to_add.items()):
        new_point = np.zeros((4, 1, nb_frame))
        new_list.append(name_point)
        new_point[0:3, 0, :] = value_point[:, :]
        new_point[3, 0, :] = 1
        new_array = np.append(new_array, new_point, axis=1)

    # Add the new points to the c3d file
    acq["parameters"]["POINT"]["LABELS"]["value"] = new_list
    acq["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = new_list.copy()

    # Some parameters need to be modified for the c3d to be working
    temp_residuals = np.zeros((1, new_array.shape[1], new_array.shape[2]))
    temp_residuals[0, : acq["data"]["meta_points"]["residuals"].shape[1], :] = acq["data"]["meta_points"]["residuals"]
    old_camera_mask = acq["data"]["meta_points"]["camera_masks"]
    temp_camera_mask = np.zeros((old_camera_mask.shape[0], new_array.shape[1], old_camera_mask.shape[2]))
    temp_camera_mask[:, :, :] = False
    temp_camera_mask[:, : acq["data"]["meta_points"]["residuals"].shape[1], :] = old_camera_mask
    acq["data"]["meta_points"]["residuals"] = temp_residuals
    acq["data"]["meta_points"]["camera_masks"] = temp_camera_mask.astype(dtype=bool)
    # Add the new analogs to the c3d file used for the type 2 platform
    acq["data"]["points"] = new_array

    return acq


if __name__ == "__main__":
    filename_origin = r".\Data\marker_less\Sujet_009\00-static-stand.c3d"
    new_filename = r"test.c3d"
    new_point_name = "mid_hip"
    # get the position in the c3d file of the R_shoulder point and the L_shoulder point
    c3d = ezc3d.c3d(filename_origin)
    point_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    R_shoulder_index = point_labels.index("R_shoulder")
    L_shoulder_index = point_labels.index("L_shoulder")
    R_shoulder_position = c3d["data"]["points"][:3, R_shoulder_index, :]
    L_shoulder_position = c3d["data"]["points"][:3, L_shoulder_index, :]

    # calculate the position of the new point as the mean of the R_shoulder and L_shoulder points, but 30 cm down in the y direction
    position = (R_shoulder_position + L_shoulder_position) / 2
    position[1] -= 300  # file in mm so 300 mm = 30 cm

    # position of the new point in the global coordinate system
    add_point_in_c3d(filename_origin, new_filename, new_point_name, position)
