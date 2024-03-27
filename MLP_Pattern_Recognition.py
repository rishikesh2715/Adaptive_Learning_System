import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Correctly list files in the directory
directory_path = os.path.expanduser("~/Documents/Adaptive_Learning_System/Data")
file_names = os.listdir(directory_path)

arrays = {}

# Iterate over each file in the directory
for file_name in file_names:
    if file_name.endswith(".txt"):
        # Correctly join the directory path and the file name to get the full path
        file_path = os.path.join(directory_path, file_name)
        # Read the array from the file
        array = np.genfromtxt(file_path)
        # Store the array in the dictionary, keying it by the file name without '.txt'
        arrays[file_name[:-4]] = array
        # print("hello")
        print(file_name[:-4])

