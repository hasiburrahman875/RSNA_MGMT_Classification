import pandas as pd
import SimpleITK as sitk
import numpy as np
import cv2
import os


def load_nifti_image(image_path):
    return sitk.ReadImage(image_path)

def create_binary_mask(image, threshold=0):
    image_array = sitk.GetArrayFromImage(image)
    print(np.unique(image_array, return_counts=True))
    binary_mask = np.zeros_like(image_array)
    binary_mask[image_array > threshold] = 1
    return binary_mask


def calculate_radiomic_features(image, mask):
    # calculate energy and volume
    image_array = sitk.GetArrayFromImage(image)
    intensity_values = image_array * mask
    energy = np.sum(intensity_values ** 2)
    volume = np.sum(mask)

    radiomic_features = {
        'energy': energy,
        'volume': volume
    }
    return radiomic_features

# Function to manually input Energy_Value and Volume for each BraTS21ID
def input_features(data):
    ids_to_discard = []  # List to store IDs to discard
    for i, row in data.iterrows():
        print("Processing BraTS21ID:", row['BraTS21ID'])
        # path to the masked images
        image_path = "/cluster/pixstor/madrias-lab/data_mining/modified_masked_output/masked_" + str(row['BraTS21ID']) + ".nii.gz"
        if not os.path.isfile(image_path):
            ids_to_discard.append(row['BraTS21ID'])
            continue
        image = load_nifti_image(image_path)
        mask = create_binary_mask(image, threshold=0)
        features = calculate_radiomic_features(image, mask)
        data.at[i, 'Energy_Value'] = features['energy']
        data.at[i, 'Volume'] = features['volume']
    # Drop rows with IDs to discard
    data = data[~data['BraTS21ID'].isin(ids_to_discard)]
    return data

# Read the train_labels CSV file
input_file = "train_labels.csv"
data = pd.read_csv(input_file)

# Manually input Energy_Value and Volume for each BraTS21ID
data = input_features(data)

# Save the results to a new CSV file
output_file = "bratsRadiomicData_on_585_samples.csv"  # Replace with desired output filename
data.to_csv(output_file, index=False)

print("New CSV file created with manually input Energy_Value and Volume for each BraTS21ID.")
