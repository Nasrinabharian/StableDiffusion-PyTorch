import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Define the directory containing the .mat files
folder_path = '/Users/abharian/Downloads/3D_Dataset/All/'

# List all files in the directory
files = os.listdir(folder_path)

data = []
percentages = []

# Loop through all files in the directory
for file_name in files:
    # Check if the file has a .mat extension
    if file_name.endswith('.mat'):
        # Create the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Load the .mat file
        mat = scipy.io.loadmat(file_path) 
        image_data = mat['Segmented_intensity']

        # Define the color to count (assuming grayscale, where color is an intensity value)
        Fat_specific_color = 1
        Fibro_specific_color = 2  

        # Count the nu5.000000-Dynamic-3dfgre-21847mber of pixels with the specific color in each slice
        Fat_counts = []
        Fibro_counts = []

        for i in range(image_data.shape[0]):  # Assuming slices are along the first dimension
            slice_data = image_data[i, :, :]
            #print(slice_data)
            count = np.sum(slice_data == Fat_specific_color)
            Fat_counts.append(count)
            count1 = np.sum(slice_data == Fibro_specific_color)
            Fibro_counts.append(count1)
            #print(f'Slice {i}: {count} pixels with color {specific_color}')
            

   

        # Total count across all slices
        total_fat_count = np.sum(Fat_counts)
        total_fibro_count = np.sum(Fibro_counts)
        fibro_over_fat = total_fibro_count / (total_fibro_count + total_fat_count)
        print(f'percentage: {fibro_over_fat}')
        fibro_over_fat = fibro_over_fat * 100
        percentages.append(fibro_over_fat)

        breast_class = 0
        if fibro_over_fat < 13.97: #One-third value: 13.96468425585974
            breast_class = 1
            data.append(1)
        elif fibro_over_fat < 25.65: #Two-thirds value: 25.649617372070704
            breast_class = 2
            data.append(2)
        else:
            breast_class = 3
            data.append(3)

        output_data = {
            **mat,  
            'fibro_over_fat': fibro_over_fat,
            'breast_class': breast_class
        }

        scipy.io.savemat('/Users/abharian/Downloads/3D_Dataset/train/'+ str(breast_class) + '/' + str(file_name) , output_data)



data_sorted = sorted(percentages)

# Calculate one-third and two-thirds indices
one_third_index = len(data_sorted) // 3
two_third_index = 2 * len(data_sorted) // 3

# Get the elements at one-third and two-thirds positions
one_third_value = data_sorted[one_third_index]
two_third_value = data_sorted[two_third_index]

# Print the results
print(f"One-third value: {one_third_value}")
print(f"Two-thirds value: {two_third_value}")




# Plotting the histogram
plt.hist(data, bins=3, edgecolor='black')

# Adding titles and labels
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()