# CaroToNet

This repository contains: 
1. The code that was used to train CaroToNet.
2. The trained model CaroToNet.pth. It was trained with all cross-sections from the training set.
3. The code to run the inference on a dataset of 2D Nifti files
4. The code to evaluate the results.
5. A MeVisLab Network to create the challenge test set used in "Learning Carotid Vessel Wall Segmentation in Blackblood MRI using Sparsely Sampled Cross-Sections from 3D Data" 

This repository does not contain the training data used in "Learning Carotid Vessel Wall Segmentation in Blackblood MRI using Sparsely Sampled Cross-Sections from 3D Data" as the subjects did not consent to publication of the data.

### Installing dependencies
This project was developed with Python 3.10.

To install dependencies run ```pip install -r requirements.txt```

# Reproducing the evaluation on the 2021 Carotid Artery Vessel Wall Segmentation Challenge[[2]](#2) test set

### Load Challenge test set and preprocess label masks, images and distance maps.

1. Get challenge test set
   1. Download from: https://vessel-wall-segmentation.grand-challenge.org/results/
   2. Unzip
2. Download and Install MeVisLab 3.7.2 from this page: https://www.mevislab.de/download 
3. Open MeVisLab
4. Add relavant MevisLabModules
   1. Go to Edit -> Preferences -> Packages
   2. Click Add Existing User Packages... 
   3. Choose the folder <path_to_this_repo>/MeVisLab
   4. Click OK
5. Install python dependency for MeVisLab
   1. Search and add the PythonPip Module![PythonPip.png](ReadMeFigures%2FPythonPip.png)
   2. Double-click on the module
   3. Install nibabel ![Nibabel.png](ReadMeFigures%2FNibabel.png)
5. Open <path_to_this_repo>/MeVisLab/create_care_ii_dataset.mlab
6. Open Care_II dataset
   1. Double-click on the CareIIDatset Module
   2. Click Open
   3. Open the complete test data folder named "testdata_withcontours_corrected" (https://vessel-wall-segmentation.grand-challenge.org/results/)
   4. Validate The dataset was loaded by clicking on the outputs of the CareIIDatset Module
7. Create preprocessed images
   1. Double-click on the run python script Module
   2. Change line 4 to your output path. Your output path is now referred to by your_output_path.
   3. Click Execute
   4. Wait until all 4189 contours are processed

### Run Inference 
````python run_inference.py -i <your_output_path>/images -m <path_to_repo>/CaroToNet.pth -o <your_output_path>/prediciton````

### Run Evaluation
````python evaluation.py -p <your_output_path>/prediciton -s <your_output_path> -gt <your_output_path>/labels````

### View Evaluation

The results of the evaluation can be seen in <your_output_path>/results.xlsx


# Training
To train the models the training data needs to be in the dataset format nnU-Net uses [[1]](#1). Channel 0000 contains the Multi Planar Reconstruction; Channel 0001 contains the distance to the centerline. 

The training can be started with:

```python train.py -d <path to the training data> -o <output path>```

# Running inference
To run the inference the data also need to be in the dataset format nnU-Net uses [[1]](#1). Channel 0000 contains the Multi Planar Reconstruction; Channel 0001 contains the distance to the centerline.

To run the inference use:

```
python run_inference.py -i <path to test images> -m <path to CaroToNet.pth> -o <output folder for predictions> -s <slice modifier explained below>
```

The slice modifier allows to only run the inference on some of the images. E.g. "*" runs the inference on all cross-sections and "*plane1*" runs the inference on all cross-sections at plane 1. 

# Running evaluation

To run the evaluation two folders must be provided that contain nifti files with the same names.

To run the evaluation use:
```
python evaluate.py -p <path of folder with predictions> -s <path to save the evaluation result> -gt <path of folder with ground truth labels> -n <if there should be a evaluation by planes enter the number of planes>
```



# References
<a id="1">[1]</a> 
Isensee,F. **nnU-Net dataset format** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md, (Accessed: 31.01.2024).

<a id="2">[2]</a> 
 “Carotid artery vessel wall segmentation challenge.” https://vessel-wall-segmentation.grand-challenge.org/. (Accessed: 31.01.2024).


# License 
The code in this repository is licensed under the GNU-GPL license.
The neural network model CaroToNet.pth is licensed under the CC BY-NC-SA 4.0 License.
(https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)