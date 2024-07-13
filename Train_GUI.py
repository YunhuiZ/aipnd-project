import tkinter as tk
from tkinter import filedialog
import subprocess
import os

# Function to run the training script with provided inputs
def run_training():
    data_dir = data_dir_entry.get()
    save_dir = save_dir_entry.get()
    arch = arch_entry.get()
    learning_rate = learning_rate_entry.get()
    hidden_units = hidden_units_entry.get()
    epochs = epochs_entry.get()
    gpu = gpu_var.get()

    command = [
        "python", "train.py", data_dir,
        "--save_dir", save_dir,
        "--arch", arch,
        "--learning_rate", learning_rate,
        "--hidden_units", hidden_units,
        "--epochs", epochs
    ]

    if gpu:
        command.append("--gpu")

    # Run the command in a subprocess
    subprocess.run(command)

# Function to browse and select a directory
def browse_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)

# Create the main window
root = tk.Tk()
root.title("Train Model")

# Create and place the input fields and labels
tk.Label(root, text="Data Directory").grid(row=0)
tk.Label(root, text="Save Directory").grid(row=1)
tk.Label(root, text="Architecture").grid(row=2)
tk.Label(root, text="Learning Rate").grid(row=3)
tk.Label(root, text="Hidden Units").grid(row=4)
tk.Label(root, text="Epochs").grid(row=5)
tk.Label(root, text="Use GPU").grid(row=6)

data_dir_entry = tk.Entry(root)
save_dir_entry = tk.Entry(root)
arch_entry = tk.Entry(root)
learning_rate_entry = tk.Entry(root)
hidden_units_entry = tk.Entry(root)
epochs_entry = tk.Entry(root)

data_dir_entry.grid(row=0, column=1)
save_dir_entry.grid(row=1, column=1)
arch_entry.grid(row=2, column=1)
learning_rate_entry.grid(row=3, column=1)
hidden_units_entry.grid(row=4, column=1)
epochs_entry.grid(row=5, column=1)

gpu_var = tk.BooleanVar()
tk.Checkbutton(root, variable=gpu_var).grid(row=6, column=1)

tk.Button(root, text="Browse", command=lambda: browse_directory(data_dir_entry)).grid(row=0, column=2)
tk.Button(root, text="Browse", command=lambda: browse_directory(save_dir_entry)).grid(row=1, column=2)
tk.Button(root, text="Start Training", command=run_training).grid(row=7, column=1)

# Set default values
arch_entry.insert(0, "vgg16")
learning_rate_entry.insert(0, "0.001")
hidden_units_entry.insert(0, "4096")
epochs_entry.insert(0, "3")

# Run the GUI loop
root.mainloop()
