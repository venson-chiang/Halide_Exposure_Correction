# Halide_Exposure_Correction
Exposure Correction using Halide

# Requirements
1.Halide 12.0.0 or above: https://github.com/halide/Halide

2.If you want to run load_weights and read model.mat in Matlab:

Deep Learning ToolBox is required: https://www.mathworks.com/products/deep-learning.html

# Methods
1.Resize input image to get low resolution image for deep learning.

2.Apply laplacian pyramid and deep learning model to correct exposure of image.

3.Apply Bilateral Guided Upsampling to get high resolution exposure corrected image. 

4.Fuse input image and exposure corrected image to get better performance of exposure correction.

# Input Images
Input images are reference from https://github.com/mahmoudnafifi/Exposure_Correction/tree/master/example_images

1.Low Exposure Images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/Rodrigo%20Valla%20-%20CC%20BY-NC%202.0.jpg" width="30%" height="30%"> <img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/a1359-NKIM_MG_6126_N1.5.JPG" width="45%" height="45%"> 

2.High Exposure Images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/a1475-dgw_146_P1.JPG" width="45%" height="45%"> <img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/example_images/a0716-MB_20030906_030_P1.5.JPG" width="45%" height="45%"> 

# Result
1.Exposure Correction of low exposure images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/Rodrigo%20Valla%20-%20CC%20BY-NC%202.0_exposure_correct.jpg" width="30%" height="30%"> <img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/a1359-NKIM_MG_6126_N1.5_exposure_correct.jpg" width="45%" height="45%"> 

2.Exposure Correction of high exposure images

<img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/a1475-dgw_146_P1_exposure_correct.jpg" width="45%" height="45%"> <img src="https://github.com/venson-chiang/Halide_Exposure_Correction/blob/main/output_images/a0716-MB_20030906_030_P1.5_exposure_correct.jpg" width="45%" height="45%"> 

More exposure correct images are in https://github.com/venson-chiang/Halide_Exposure_Correction/tree/main/output_images

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
Exposure correction model is reference to https://github.com/mahmoudnafifi/Exposure_Correction

Bilateral Guided Upsampling is reference to https://github.com/google/bgu

