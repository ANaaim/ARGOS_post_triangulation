from tkinter import Tk, filedialog


def choose_file_npy():
    """
    Open a file dialog to choose a .npy file.

    Returns
    -------
    filename : str
        The path to the chosen file.
    """
    root = Tk()
    root.withdraw()  # cache la fenêtre principale

    filename = filedialog.askopenfilename(
        title="Choisir un fichier NPY", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )

    root.destroy()
    return filename
