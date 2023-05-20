import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from dataclasses import dataclass

 # Initialising root class for the app, which inherits from tkinter.Tk
class Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2-class PCA")
        self.frame = tk.Frame(self)
        self.frame.pack()
        self.create_widgets()
    
    # Calls all methods for generating widgets
    def create_widgets(self):
        # dummy variable to dictate grid positions
        self.i = 0

        self.user_info()
        self.data()
        self.method()
        self.plots()
        self.save_dir()
        self.enter_button()

    def user_info(self):
        self.user_info_frame = tk.LabelFrame(self.frame, text="User Information")
        self.user_info_frame.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

        self.name_label = tk.Label(self.user_info_frame, text="Enter your name: ")
        self.name_label.grid(row=0, column=0, sticky="news", padx=(10, 0), pady=10)
        self.name_entry = tk.Entry(self.user_info_frame)
        self.name_entry.grid(row=0, column=1, sticky="news", padx=10, pady=10)

    def data(self):
        self.data_frame = tk.LabelFrame(self.frame, text="Data Entry")
        self.data_frame.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

        # Filepath entry
        self.filepath_label = tk.Label(self.data_frame, text="Data Filepath")
        self.filepath_label.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=10)
        self.filepath_entry = tk.Entry(self.data_frame, width=80)
        self.filepath_entry.grid(row=0, column=1, padx=10, pady=10)
        self.filepath_button =tk.Button(self.data_frame, text="Select", command=self.get_filepath)
        self.filepath_button.grid(row=0, column=2, sticky="news", padx=(0, 10), pady=10)
    
        # Scaling method entry
        self.scaling_label = tk.Label(self.data_frame, text="Scaling Method")
        self.scaling_label.grid(row=1, column=0, sticky="w", padx=(10, 0), pady=10)
        self.scaling_entry = ttk.Combobox(self.data_frame, state="readonly", values=["Standard", "Pareto", "None"])
        self.scaling_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=10)

        # Significance level entry
        self.sig_level_label = tk.Label(self.data_frame, text="Significance Level")
        self.sig_level_label.grid(row=2, column=0, sticky="w", padx=(10, 0), pady=10)
        self.sig_level_entry = tk.Spinbox(self.data_frame, from_=0, to=1, increment=0.01)
        self.sig_level_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=10)
    
    def method(self):
        self.method_frame = tk.LabelFrame(self.frame, text="Choose ML Method")
        self.method_frame.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

        self.method_label = tk.Label(self.method_frame, text="Method: ")
        self.method_label.grid(row=0, column=0, sticky="news", padx=(10, 0), pady=10)
        self.method_entry = ttk.Combobox(self.method_frame, state="readonly", values=["PCA", "PLS-DA", "OPLS-DA", "SVM", "RF"])
        self.method_entry.grid(row=0, column=1, sticky="news", padx=(10, 0), pady=10)
    
    def get_filepath(self):
        # Delete current text in Entry box
        self.filepath_entry.delete(0, tk.END)

        # Open directory to choose filepath
        self.filepath = filedialog.askopenfilename(title="Select input data", filetypes=(("csv files", "*.csv"), ("All files", "*.*")))
        
        # Paste filepath text into Entry box
        self.filepath_entry.insert(tk.END, self.filepath)

    def plots(self):
        self.plots_frame = tk.LabelFrame(self.frame, text="Plotting")
        self.plots_frame.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

        # Control/case checks
        self.control_label = tk.Label(self.plots_frame, text="Enter control: ")
        self.control_label.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=10)
        self.control_entry = tk.Entry(self.plots_frame)
        self.control_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=10)

        self.case_label = tk.Label(self.plots_frame, text="Enter case: ")
        self.case_label.grid(row=0, column=2, sticky="w", padx=(10, 0), pady=10)
        self.case_entry = tk.Entry(self.plots_frame)
        self.case_entry.grid(row=0, column=3, sticky="w", padx=(10, 0), pady=10)

        # Choosing plot colours
        self.control_colour_label = tk.Label(self.plots_frame, text="Control plot colour: ")
        self.control_colour_label.grid(row=1, column=0, sticky="w", padx=(10, 0), pady=10)
        self.control_colour_entry = ttk.Combobox(self.plots_frame, state="readonly", values=["Green", "Blue", "Red", "Yellow", "Purple", "Orange", "Cyan", "Magenta", "Black"])
        self.control_colour_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=10)
        self.case_colour_label = tk.Label(self.plots_frame, text="Case plot colour: ")
        self.case_colour_label.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=10)
        self.case_colour_entry = ttk.Combobox(self.plots_frame, state="readonly", values=["Green", "Blue", "Red", "Yellow", "Purple", "Orange", "Cyan", "Magenta", "Black"])
        self.case_colour_entry.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=10)

    def save_dir(self):
        self.save_dir_frame = tk.LabelFrame(self.frame, text="Save")
        self.save_dir_frame.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

        # Save directory entry
        self.save_dir_label = tk.Label(self.save_dir_frame, text="Save Here")
        self.save_dir_label.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=10)
        self.save_dir_entry = tk.Entry(self.save_dir_frame, width=80)
        self.save_dir_entry.grid(row=0, column=1, padx=10, pady=10)
        self.save_dir_button = tk.Button(self.save_dir_frame, text="Select", command=self.get_save_dir)
        self.save_dir_button.grid(row=0, column=2, sticky="news", padx=(0, 10), pady=10)
    
    def get_save_dir(self):
        # Delete current text in Entry box
        self.save_dir_entry.delete(0, tk.END)

        # Open directory to choose filepath
        self.save_dir_path = filedialog.askdirectory(title="Select save location")

        # Paste filepath text into Entry box
        self.save_dir_entry.insert(tk.END, self.save_dir_path)

    def enter_button(self):
        self.enter = tk.Button(self.frame, text="Enter data", command=self.enter_data)
        self.enter.grid(row=self.i, column=0, sticky="news", padx=20, pady=20)
        self.i += 1

    def enter_data(self):
        self.name = self.name_entry.get()
        self.filepath = self.filepath_entry.get()
        self.scaling = self.scaling_entry.get()
        self.q = self.sig_level_entry.get()
        self.control = self.control_entry.get()
        self.case = self.case_entry.get()
        self.control_colour = self.control_colour_entry.get().lower()
        self.case_colour = self.case_colour_entry.get().lower()
        self.save_dir_path = self.save_dir_entry.get()

        # Create output attribute to be used in PCA
        self.guidata = GUIData(
            name=self.name,
            filepath=self.filepath,
            scaling=self.scaling,
            q=float(self.q),
            control=self.control,
            case=self.case,
            control_colour=self.control_colour,
            case_colour=self.case_colour,
            save_dir_path=self.save_dir_path
        )

        ### Check for errors

        # Close app after data has been entered
        self.quit()

@dataclass
class GUIData():
    """
    Class for tracking output of the GUI.
    """
    name: str
    filepath: str
    scaling: str
    q: float
    control: str
    case: str
    control_colour: str
    case_colour: str
    save_dir_path: str

if __name__ == '__main__':
    app = Window()
    app.mainloop()
