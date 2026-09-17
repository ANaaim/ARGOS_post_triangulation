from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
import numpy as np
from biobuddy import C3dData
import biorbd
from tkinter import Tk, filedialog
from pathlib import Path




def choose_file():
    root = Tk()
    root.withdraw()  # cache la fenêtre principale

    filename = filedialog.askopenfilename(
        title="Choisir un fichier NPY",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )

    root.destroy()
    return filename


def visualize(npy_file):

    data = np.load(npy_file, allow_pickle=True).item()

    q = data["q"]
    model_name = data["model_path"]
    filename = data["filename_path"]
    # to avoid error in Pyorerun this line was necessary
    model_name = model_name.replace("\\", "/")

    model = biorbd.Model(model_name)
    trial = C3dData(filename)
    print(model_name)
    print(filename)
    print(model_name)
    print(type(model_name))

    markerNames = [
        model.markerNames()[i].to_string()
        for i in range(len(model.markerNames()))
    ]
    markers = trial.get_position(markerNames)[:3, :, :]

    nb_frames = q.shape[1]
    t_span = np.linspace(0, 10, nb_frames)

    model_rerun = BiorbdModel(model_name)
    
    print("Model path:", model_name)
    print("Model stem:", Path(model_name).stem)

    viz = PhaseRerun(t_span)
    viz.add_animated_model(
        model_rerun,
        q,
        tracked_markers=PyoMarkers(
            data=markers,
            channels=markerNames,
            show_labels=False,
        ),
    )

    viz.rerun()


if __name__ == "__main__":

    import sys

    if len(sys.argv) >1:
        npy_file = sys.argv[1]
    else:
        npy_file = choose_file()

    if npy_file:
        visualize(npy_file)
