from pathlib import Path
from pre_process import correct_markerless_fq_folder

# defintion of the folder of the data 
DATA_FOLDER = Path("data/")
raw_data_folder = DATA_FOLDER / "raw"
marker_less_model = ["Synthpose"]

# pre_processed data folder
pre_processed_data_folder = DATA_FOLDER / "pre_processed"
pp_marker_based_folder = pre_processed_data_folder / "marker_based"
pp_marker_less_folder = pre_processed_data_folder / "marker_less"

# file containing the cut of the marker-based data


def pre_process_marker_based_data():
    """
    Pre-process the marker-based data and save it to the pre-processed folder.
    """
    # Trim the marker-based data based on the file 
    pass

def pre_process_marker_less_data():
    """
    Pre-process the marker-less data and save it to the pre-processed folder.
    """
    # for each marker-less model, pre-process the data and save it to the pre-processed folder.
    # take into account that it is possible that there is only one marker-less model, but the code should be able to handle multiple models.
    for model in marker_less_model:
        # Pre-process the data for each model
        # Correction of the frame rate of the marker-less data based on the corresponding marker-based data.
        folder_markerless_to_correct = raw_data_folder / model
        folder_marker_based = raw_data_folder / "marker_based"
        folder_marker_less_fq_corrected = pp_marker_less_folder /"ML_fq_corrected" / model
        correct_markerless_fq_folder(folder_marker_based, folder_markerless_to_correct, folder_marker_less_fq_corrected)
        # trim the marker-less data based on the cut of the marker-based data
        #trim_marker_less_data(folder_marker_less_fq_corrected,)        

if __name__ == "__main__":
    pre_process_marker_based_data()
    pre_process_marker_less_data()


