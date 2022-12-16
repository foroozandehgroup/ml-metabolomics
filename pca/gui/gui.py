import tkinter as tk
from tkinter import ttk

 
# Initialising main window and frame for GUI
window = tk.Tk()
window.title("2-class PCA")

frame = tk.Frame(window)
frame.pack()


# User information
user_info_frame = tk.LabelFrame(frame, text="User Information")
user_info_frame.grid(row=0, column=0, padx=20, pady=20)

full_name_label = tk.Label(user_info_frame, text="Full Name")
full_name_label.grid(row=0, column=0)
full_name_entry = tk.Entry(user_info_frame)
full_name_entry.grid(row=1, column=0)


# Data information
data_frame = tk.LabelFrame(frame, text=" ")


window.mainloop()