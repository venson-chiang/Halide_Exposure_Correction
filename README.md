# Halide_Exposure_Correction
Exposure Correction using Halide

# Requirements
Halide 12.0.0 or above: https://github.com/halide/Halide

If you want to run load_weights and read model.mat
Matlab machine learning toolkit is needed.

# Input Images
Input images are reference from https://github.com/mahmoudnafifi/Exposure_Correction/tree/master/example_images

1.Low Exposure Images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/Rodrigo%20Valla%20-%20CC%20BY-NC%202.0.jpg" width="50%" height="50%"> 

2.High Exposure Images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/a1475-dgw_146_P1.JPG" width="50%" height="50%"> 

# Result
1.Exposure Correction for low exposure image

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/Rodrigo%20Valla%20-%20CC%20BY-NC%202.0_exposure_correct.jpg" width="50%" height="50%"> 

2.Exposure Correction for high exposure image

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/a1475-dgw_146_P1_exposure_correct.jpg" width="50%" height="50%"> 




# Usage
1. Change HALIDE_DISTRIB_PATH to yours in Makefile.inc
```
HALIDE_DISTRIB_PATH ?= /mnt/d/Software/Halide-12/distrib 
```
2. Run Makefile 
```
make test
```


# Reference
The method used in this project is reference to https://github.com/mahmoudnafifi/Exposure_Correction

