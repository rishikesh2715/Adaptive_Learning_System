import os
import numpy as np

directory_path = os.path.expanduser("~/Documents/Adaptive_Learning_System/Data/")
file_names = os.listdir(directory_path)

features = []

for file_name in file_names:
    if file_name.endswith(".txt"):
        file_path = os.path.join(directory_path, file_name)
        array = np.genfromtxt(file_path, dtype='int')
        features.append(array)
        # print(array)
        print(array.shape)
features = np.array(features)
# print(features)

