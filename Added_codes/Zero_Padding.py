import scipy.io as sio
import numpy as np
import os

# Specify the directory containing the .mat files
directory = '/Users/abharian/Downloads/3D_Dataset/train/3'
destination_directory = '/Users/abharian/Downloads/3D_Dataset/train/Padded_3'
# Desired shape
desired_shape = (511, 351, 119)

def pad_image_to_shape(image, target_shape):
    """Pad the 3D image with zeros to match the target shape."""
    padded_image = np.zeros(target_shape)
    
    # Determine the dimensions to be used for copying
    x_min = min(image.shape[0], target_shape[0])
    y_min = min(image.shape[1], target_shape[1])
    z_min = min(image.shape[2], target_shape[2])
    
    # Copy the data
    padded_image[:x_min, :y_min, :z_min] = image[:x_min, :y_min, :z_min]
    
    return padded_image

# Iterate over all .mat files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory, filename)
        
        # Load the .mat file
        mat_data = sio.loadmat(file_path)
        
        # Check if 'Segmented_intensity' exists in the file
        if 'Segmented_intensity' in mat_data:
            segmented_intensity = mat_data['Segmented_intensity']
            
            # Pad the image to the desired shape
            padded_image = pad_image_to_shape(segmented_intensity, desired_shape)
            
            # Save the padded image back to a new .mat file
            new_filename = f'{filename}'
            new_file_path = os.path.join(destination_directory, new_filename)
            sio.savemat(new_file_path, {'Segmented_intensity': padded_image})
            
            print(f"Processed and saved {new_filename}")
        else:
            print(f"Segmented_intensity not found in {filename}")

print("All files processed.")
