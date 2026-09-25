from tkinter import Tk, filedialog


def choose_file():
    root = Tk()
    root.withdraw()  # cache la fenêtre principale

    filename = filedialog.askopenfilename(
        title="Choisir un fichier NPY", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )

    root.destroy()
    return filename
