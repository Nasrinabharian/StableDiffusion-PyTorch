import os
import scipy.io
import numpy as np

def get_max_shape(directory):
    max_shape = (0, 0, 0)  # Initialize with the smallest possible shape
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            # Load the .mat file
            mat_data = scipy.io.loadmat(os.path.join(directory, filename))
            
            # Access the 'Segmented_intensity' data
            if 'Segmented_intensity' in mat_data:
                data = mat_data['Segmented_intensity']
                
                # Get the shape of the 3D data
                current_shape = data.shape
                
                # Determine the max shape
                max_shape = tuple(max(max_shape[i], current_shape[i]) for i in range(3))
    
    return max_shape

# Specify the directory containing the .mat files
directory = '/Users/abharian/Downloads/3D_Dataset/train/3'

# Get the maximum shape
max_shape = get_max_shape(directory)

print(f"The maximum shape of 'Segmented_intensity' across all .mat files is: {max_shape}")
