# Diffusing Coupling High-Frequency-Purifying Structure Feature Extraction for Brain Multimodal Registration

This is the code for the BIBM2023 paper Diffusing Coupling High-Frequency-Purifying Structure Feature Extraction for Brain Multimodal Registration.

:fire: **Our paper is accepted by BIBM 2023 as regular paper** :fire:  
## Abstract of The Paper

The core of medical image registration is the alignment of corresponding structures.
However, in multimodal image registration, substantial differences in appearance (intensity distribution) of the images often compel the registration model to prioritize intensity information over structure information, resulting in low accuracy of registration. 
Therefore, the disentangling structure information from intensity information is vital to improve the registration effectiveness. 
To this end, we propose a diffusing coupling high-frequency-purifying structure feature extraction for brain multimodal registration. 
Specifically, the denoising diffusion probabilistic models (DDPM) is firstly utilized to extract complete feature information from images. 
Then, the discrete cosine transform (DCT) is applied to purify high-frequency structure information from the complete feature information for registration.
Furthermore, structure consistency constraint (SCC) is introduced based on purified structure information to emphasize the core position of the structure in registration.
Through comprehensive comparisons with traditional and learning-based methods on the multimodal brain MRI dataset, our method demonstrates superior accuracy and stability in brain multimodal registration. 
The anonymous code is available at https://github.com/goghfan/DDNet. 

## How to Train

### Data part

We used datasets from OASIS and ASDI, which are available at the URL below.
https://www.oasis-brains.org/#access
https://adni.loni.usc.edu/
We have gone through a series of preprocessing, the specific measures are as follows:

1. Remove the skull: You can remove the skull data through the Freesurfer software or directly use the part that comes with the data set.
2. Image cropping: crop and resample the data to [128 * 128 * 128]
3. Image intensity normalized to [-1,1]


### Train Part

You can train the entire model by running train.py, and the dependent libraries can be found in requirements.txt.
We recommend that you use GPU with more than 12GB of memory.


## The Function of  Our Code

**1. ddpm**

The basic model used by DDPM is stored in this folder. 

**2. DDPM_train**

**2.1 data**

The data collection and processing methods used by DDPM are stored in this folder.

**2.2 ddpm**

The basic model used by DDPM is stored in this folder. 

**3. registration**

In this folder, all files used to generate registration results are stored.

**4. voxelmorph**

The files under this folder store the framework of the improved voxelmorph used for our registration method.

**5. voxelmorph_baseline**

The files in this folder store the framework of the basic voxelmorph used for our registration method.

**6. metric**

Files for processing data and testing registration effects are stored in this folder.

**7. train_DCT_MSE.py train.py vm_train.py**

This is part of the files that train our ensemble method.
## The Pre-trained Model will be updated soon.
