import numpy as np


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

