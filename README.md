This is a project by team **MediMiners.** 
Team Members: _Ganesh, Hasibur, Mizanur, and Sazed (Order is insignificant)_

### Dataset Link
https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/

It contains all the images and `train_labels.csv`

### Environment for Densenet and Data Preprocessing
Please install the required packages using _pip install -r requirement.txt_ in the corresponding conda environment. 

### DICOM to NIFTII
To convert the DICOM format to NIFTII format, please use `dicomtoniftii.ipynb` under the `preprocessing` folder.

### Densent 169
For central preprocessing and interval preprocessing, please utilize the corresponding codes. Please select the exact MRI type and provide the accurate data path for smooth execution. The results will be saved in a CSV file for further evaluation and ensembling. 

### Radiomic Part
the conda environment for the radiomic feature code is radiomic.yml

you can download default vit_h from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

For Radiomic Feature Code run these files sequentially<br/>
1. `3dMaskGen.py`<br/>
2. `radDataCreation.py`<br/>
3. `radiomicClassifier.py`<br/>

Note: ensure that all the data paths in those file is accurate. The code is done by using absolute path

### Mask Generation using Segment Anything Model (SAM)
The 3D NIFTII mask for the 3D NIFTII image is generated using script `3dMaskGen.py`. Ensure  the input directory of the 3D NIFTII images, location of the vit_h of the pretrained SAM and output directory for masks are updated as absolute path has been given there.

### Radiomic Feature Creation from 3D NIFTII mask
For volume and energy radiomic feature has been extracted from the 3D mask using the script `radDataCreation.py`. Ensure  the input directory of the 3D NIFTII masks, location of the `train_labels.csv` and output csv `bratsRadiomicData_on_585_samples.csv` location is updated.

### MGMT value prediction
`radiomicClassifier.py` script is used to predict the MGMT value from the `radbratsRadiomicData_on_585_samples.csv`. This script also generates the train and test set and there prediction probabilities.


### Ensemble and ROC Curve
The ROC curve is generated differently in the `dm_auc_generation.ipynb` file. The ROC curve generator takes the predicted values for different models or MRI types, along with their maximum, minimum, and average values, which are provided in the corresponding CSV files in the `Prediction_CSV` folder.