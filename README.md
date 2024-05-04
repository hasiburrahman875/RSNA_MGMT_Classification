This is a project by team **MediMiners.** 
Team Members: _Ganesh, Hasibur, Mizanur, and Sazed (Order is insignificant)_

### Dataset Link
https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/

### Environment for Densenet and Data Preprocessing
Please install the required packages using _pip install -r requirement.txt_ in the corresponding conda environment. 

### DICOM to NIFTII
To convert the DICOM format to NIFTII format, please use `dicomtoniftii.ipynb` under the `preprocessing` folder.

### Densent 169
For central preprocessing and interval preprocessing, please utilize the corresponding codes. Please select the exact MRI type and provide the accurate data path for smooth execution. The results will be saved in a CSV file for further evaluation and ensembling. 

### Ensemble and ROC Curve
The ROC curve is generated differently in the `dm_auc_generation.ipynb` file. The ROC curve generator takes the predicted values for different models or MRI types, along with their maximum, minimum, and average values, which are provided in the corresponding CSV files in the `Prediction_CSV` folder.


### Radiomic Feature
the conda environment for the radiomic feature code is radiomic.yml

you can download default vit_h from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

For Radiomic Feature Code run these files sequentially<br/>
1. 3dMaskGen.py <br/>
2. radDataCreation.py<br/>
3. radiomicClassifierUpdated.py<br/>

Note: ensure that all the data paths in those file is accurate. The code is done by using absolute path
