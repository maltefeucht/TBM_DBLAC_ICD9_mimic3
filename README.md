## TBM_ICD9_mimic3
This contains the code for the masterthesis with the title "Classification of ICD-9 Codes from Unstructured Clinical Notes using Transformer-Based Neural Networks".

### Setup
- This project is developed in python 3.8
- Install dependencies using the provided requirements.txt. Other package versions might work as well, but it is recommended to install the package versions as specified in the requirements.txt.

### Preprocessing
- The preprocessing is based on the preprocessing of the the [CAML](https://arxiv.org/abs/1802.05695) model architecture proposed by Mullenbach et al.
- First, edit the local and remote DATA_DIR, MIMIC_3_DIR and PROJECT_DIR in constants_mimic3.py to make them point to your respective data directories.
- Organize the data with the following structure:

mimicdata\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D_ICD_DIAGNOSES.csv\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D_ICD_PROCEDURES.csv\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ICD9_descriptions (already in repo)\
|&ndash;&ndash;&ndash;mimic3\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NOTEEVENTS.csv\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DIAGNOSES_ICD.csv\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PROCEDURES_ICD.csv\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*_hadm_ids.csv (already in repo)

Obtain the MIMIC-III files here: https://physionet.org/content/mimiciii/1.4/

- Run dataproc_mimic_III.ipynb. This might take a while.
- If you are curious, after running dataproc_mimic_III.ipynb you can run data_visualization.ipynb to get plots and statistics on the MIMIC-III and the MIMIC-III-50 dataset.

### Training
- To train one of the provided models, run sh train_<model_name>.sh in the scripts directory of the respective model directory.

### Testing
- To test and reproduce the results for one of the models, run sh test_<model_name>.sh in the scripts directory of the respective model directory. This will load the best performing model obtained over k-fold split training for testing. 

### Inference
- To run inference and get predictions for one of the models, run sh inference_<model_name>.sh in the scripts directory of the respective model directory. This will load the best performing model obtained over k-fold split training for inference. The predictions are stored in the results directory. 