import subprocess
import os

# List of folder names to be created
folders = ["class", "img", "log", "model", "raw"]
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))

# Iterate over the folder names
for folder in folders:
    # Define the folder path
    folder_path = os.path.join(parent_directory, folder)

    # Use subprocess to run the mkdir command
    subprocess.run(["mkdir", folder_path], shell=True)
