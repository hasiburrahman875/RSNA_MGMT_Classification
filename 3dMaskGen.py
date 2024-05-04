import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os

# Load pre-trained SAM model 
# Give SAM model path, here we are using the sam_vit_h_4b8939.pth
# you can download default vit_h from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
sam_checkpoint = '/cluster/pixstor/madrias-lab/jewel/mask_generation/sam_vit_h_4b8939.pth'
model_type = "vit_h"

# device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)

def get_index_of_max_value(height_values):
    max_value = max(height_values)
    index_of_max = height_values.index(max_value)
    return index_of_max

def generate_3d_mask_slice_by_slice(nifti_file, input_dir, output_dir, sam_model):
    nifti_path = os.path.join(input_dir, nifti_file)

    # Load NIfTI image
    nifti_image = nib.load(nifti_path)
    nifti_data = nifti_image.get_fdata()

    # Initialize empty 3D mask
    mask_3d = np.zeros_like(nifti_data)

    # Process each slice
    # for slice_idx in tqdm(range(nifti_data.shape[2])):
    for slice_idx in tqdm(range(nifti_data.shape[2])):
        # Get the current 2D slice
        current_slice = nifti_data[:, :, slice_idx]

        # Convert to PyTorch tensor
        input_tensor = torch.from_numpy(current_slice).unsqueeze(0).float()  # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            # output_mask = sam_model(input_tensor, multimask_output=False)
            input_data = {"image":input_tensor,
                            "original_size":(nifti_data.shape[2], nifti_data.shape[2])}
            output_mask = sam_model([input_data], multimask_output=True)[0] # get 0 index

        # Convert output mask to numpy array
        best_mask_idx = np.argmax(output_mask['iou_predictions'].cpu().numpy())
        mask_array = output_mask['masks'].cpu().numpy()[0][best_mask_idx]
 
        # Store the 2D mask in the 3D mask
        mask_array = mask_array.astype(np.uint8)*255
        # print(np.unique(mask_array, return_counts=True))
        mask_3d[:,:, slice_idx] = mask_array

    # Save the 3D mask as a new NIfTI file
    
    output_nifti_image = nib.Nifti1Image(mask_3d, affine=np.eye(4))
    output_filename = f"masked_{nifti_file}"
    output_filepath = os.path.join(output_dir, output_filename)
    nib.save(output_nifti_image, output_filepath)

# Example usage
# nifti_path = '/cluster/pixstor/madrias-lab/data_mining/brats_classification/resized/2.nii.gz'

# Call the function to generate the 3D mask slice by slice and save it
# generate_3d_mask_slice_by_slice(nifti_path, sam_model)

# input dir path to resized 512x512x512 nifti image
input_dir = "/cluster/pixstor/madrias-lab/data_mining/brats_classification/resized"

# output dir path to resized 512x512x512 nifti image
output_dir = "/cluster/pixstor/madrias-lab/data_mining/modified_masked_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nifti_files = [f for f in os.listdir(input_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
processed_nifti_files = [f.replace('masked_', '') for f in os.listdir(output_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

for nifti_file in tqdm(nifti_files):
    if nifti_file not in processed_nifti_files:
        masked_nifti_image = generate_3d_mask_slice_by_slice(nifti_file, input_dir, output_dir, sam_model)
    else:
        print(nifti_file, "already processed and mask created")
print("mask generation completed")