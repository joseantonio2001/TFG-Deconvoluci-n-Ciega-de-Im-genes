# Introduction
Este repositorio es una extensión del repositorio original [CHSG-DDNET](https://github.com/vipgugr/CHSG-DDNet) donde se incluyen métodos para restaurar y mejorar imágenes utilizando DIP (Deep Image Prior).

# Requirements
* Python >= 3.8
* Numpy
* Scikit-image
* Pytorch == 1.7.1
* pytorch_msssim
* Scipy


# Installation
```console
conda create -n ddnet python=3.8 scipy scikit-image
conda activate ddnet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install pytorch_msssim
```

# Usage
First, activate conda environment: 
```console 
conda activate ddnet
```

## NO_DIP model

### Data preparation training
Download dataset from this [link](https://drive.google.com/drive/folders/109VwKx-GI_MbqIdAfGp6WJ0ekpA1N9_T?usp=sharing) and extract it. Modify `TRAIN_FILE_PATH` and `EVAL_FILE_PATH` in config.py to the paths of the train data folder and the validation data folder.

### Train
Our systems uses two models:
1. Luminance model or "y model". Script: train_y.py
2. Color model or "cbcr model".  Script: train_cbcr.py

Before training, modify both `W_PATH_SAVE` and `W_COLOR_PATH_SAVE` to the paths of the folders where you want to save the weights of both models.

```console
python train_y.py
python train_cbcr.py
```

### Val/Test
To process a folder and generate the restores images of its contents, use predict.py.
You will need to download our weights from [here](https://drive.google.com/drive/folders/109VwKx-GI_MbqIdAfGp6WJ0ekpA1N9_T?usp=sharing) or provide your own using the training scripts.

```console
python predict.py <image_path> <psf_path> <output_path> <model_y_weights_path> <model_cbcr_weights_path>
#python predict.py data/degraded_noisy/test2017reduced data/restored_noisy_denoise/test2017reduced/psfs/ out model_y_weights.pt model_cbcr_weights.pt
```
* <image_path>:  Path to folder containing the blur images. They have to be in png or jpg formats.
* <psf_path>:    Path to folder containing the estimated PSFs. They must be saved as a matrix in a npy file. Each one must have the same name as its corresponding blur image with the subfix `_psf.npy` added.
* <output_path>: Path where the restored images would be saved.
* <model_y_weights_path>: Path to the weights of model y.
* <model_cbcr_weights_path>: Path to the weights of model cbcr.

## DIP model

### Data preparation training
Modificar los paths del archivo `config_deep_image_prior.py` con las rutas de los directorios donde residen los datos de trabajo (`.mat`, `.npy`, `.png`)

#### Descripción de los Paths
    - `ORIGINAL_PNGs_PATH`: Directory containing the original images in PNG format.
    - `TRAIN_PNGs_PATH`: Directory with the degraded images used for training.
    - `TRAIN_NPYs_PATH`: Directory with the numpy (.npy) files for training.
    - `TRAIN_NPYs_PSFs_PATH`: Directory with the PSF (.npy) files for training.
    - `PREDICTS_PNGs_PATH`: Directory with the degraded images to be restored.
    - `PREDICTS_PSFs_MATs_PATH`: Directory with the PSF (.mat) files for restoration.
    - `PREDICTS_PSFS_PNGs_PATH`: Directory with the PSF (.png) files for restoration.
    - `W_Y_PATH`: Path of the weights of the y model
    - `W_COLOR_PATH`: Path of the weights of the cbcr model
