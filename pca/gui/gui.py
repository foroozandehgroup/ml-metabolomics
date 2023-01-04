import tkinter as tk
from tkinter import ttk
from tkinter import filedialog


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
        self.user_info()
        self.data()
        self.plots()
        self.enter_button()

    def user_info(self):
        self.user_info_frame = tk.LabelFrame(self.frame, text="User Information")
        self.user_info_frame.grid(row=0, column=0, sticky="news", padx=20, pady=20)

        self.name_label = tk.Label(self.user_info_frame, text="Enter your name: ")
        self.name_label.grid(row=0, column=0, sticky="news", padx=(10, 0), pady=10)
        self.name_entry = tk.Entry(self.user_info_frame)
        self.name_entry.grid(row=0, column=1, sticky="news", padx=10, pady=10)

    def data(self):
        self.data_frame = tk.LabelFrame(self.frame, text="Data Entry")
        self.data_frame.grid(row=1, column=0, sticky="news", padx=20, pady=20)

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
    
    def get_filepath(self):
        # Delete current text in Entry box
        self.filepath_entry.delete(0, tk.END)

        # Open directory to choose filepath
        self.filepath = filedialog.askopenfilename(title="Select input data", filetypes=(("csv files", "*.csv"), ("All files", "*.*")))
        
        # Paste filepath text into Entry box
        self.filepath_entry.insert(tk.END, self.filepath)

    def plots(self):
        self.plots_frame = tk.LabelFrame(self.frame, text="Plotting")
        self.plots_frame.grid(row=2, column=0, sticky="news", padx=20, pady=20)

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
        self.control_colour_entry = ttk.Combobox(self.plots_frame, state="readonly", values=["Green", "Blue", "Red", "Yellow"])
        self.control_colour_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=10)
        self.case_colour_label = tk.Label(self.plots_frame, text="Case plot colour: ")
        self.case_colour_label.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=10)
        self.case_colour_entry = ttk.Combobox(self.plots_frame, state="readonly", values=["Green", "Blue", "Red", "Yellow"])
        self.case_colour_entry.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=10)


    def enter_button(self):
        self.enter = tk.Button(self.frame, text="Enter data", command=self.enter_data)
        self.enter.grid(row=3, column=0, sticky="news", padx=20, pady=20)

    def enter_data(self):
        self.name = self.name_entry.get()
        self.filepath = self.filepath_entry.get()
        self.scaling = self.scaling_entry.get()
        self.q = self.sig_level_entry.get()
        self.control = self.control_entry.get()
        self.case = self.case_entry.get()
        self.control_colour = self.control_colour_entry.get().lower()
        self.case_colour = self.case_colour_entry.get().lower()

        ### Check for errors

        # Close app after data has been entered
        self.destroy()
            

if __name__ == '__main__':
    app = Window()
    app.mainloop()
