import numpy as np


def calculate_RAB(point_data: np.ndarray, name_point_list: list):
    # TODO: to make more clear
    # 1. Extract marker labels list from the C3D object
    all_labels = [label.strip() for label in name_point_list]

    n_frames = point_data.shape[2]

    # Helper function to get marker coordinates safely
    def get_marker(label_name):
        return point_data[0:4, all_labels.index(label_name), :]

    # --- 2. CALCULATE VIRTUAL MARKERS ---

    # Midpoints for ISB Thorax
    mid_end = (get_marker("IJ") + get_marker("C7")) / 2.0
    mid_start = (get_marker("PX") + get_marker("T10")) / 2.0

    Z_thorax = mid_end - mid_start
    Z_thorax /= np.linalg.norm(Z_thorax, axis=0)
    # Bi-acromial distance & Glenohumeral Joint Centers (Rab 2002)
    R_AC = get_marker("R_AC")
    L_AC = get_marker("L_AC")
    D_all = np.linalg.norm(R_AC[:3, :] - L_AC[:3, :], axis=0)
    D = np.mean(D_all)

    R_GH = np.ones((4, n_frames))
    L_GH = np.ones((4, n_frames))

    # Right shoulder center
    R_GH = R_AC.copy() - Z_thorax * (0.17 * D)

    # Left shoulder center
    L_GH = L_AC.copy() - Z_thorax * (0.17 * D)

    return R_GH, L_GH


def calculate_mid_point(point_data: np.ndarray, name_point_list: list):
    # elbow joint center
    # 1. Extract marker labels list from the C3D object
    all_labels = [label.strip() for label in name_point_list]

    n_frames = point_data.shape[2]

    # Helper function to get marker coordinates safely
    def get_marker(label_name):
        return point_data[0:4, all_labels.index(label_name), :]

    R_EJC = (get_marker("R_EL") + get_marker("R_EM")) / 2
    L_EJC = (get_marker("L_EL") + get_marker("L_EM")) / 2
    R_WJC = (get_marker("R_RS") + get_marker("R_US")) / 2
    L_WJC = (get_marker("L_RS") + get_marker("L_US")) / 2
    R_FJC = (get_marker("R_HM5") + get_marker("R_HM2")) / 2
    L_FJC = (get_marker("L_HM5") + get_marker("L_HM2")) / 2

    return R_EJC, L_EJC, R_WJC, L_WJC, R_FJC, L_FJC
