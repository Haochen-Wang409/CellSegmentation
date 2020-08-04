# CellSegmentation

This is a project that mainly uses U-Net and Watershed to do cell segmentation. (mainly use PyTorch and OpenCV)  
All the data comes from http://celltrackingchallenge.net/

There are two main subprojects. One for 2-dimention cells and the other one for 3-dimention cells.  
Because of the particularity of U-Net, the size of input images and ground truths must be a power of 2, including the number of image slices in the 3-dimentional case.

To have a better understanding of this algorithm, I recomend you to read:  
> U-Net: Convolutional Networks for Biomedical Image Segmentation (arXiv:1505.04597v1 [cs.CV] 18 May 2015)  
> 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation (arXiv:1606.06650v1 [cs.CV] 21 Jun 2016)

## Environment  
- `PyTorch 1.5.1`  
- `Tensorflow 2.2.0` (to match TensorboardX)  
- `Cudatoolkit 10.1` (depend on your CUDA version)  
- `Tensorboard 2.2.2`  
- `TensorboardX 2.1`  
- `OpenCV-Python 4.3.0.36`  

Other versions might work as well but I recomend you to install the version above.  
In the 3-dimentional case, `SimpleITK` and `libtiff` are required as well.

## Steps
The algorithm mainly contains 4 steps.  
> Step 1. Data Augmentation (Flip, Crop etc.)  
> Step 2. Preprocessing (Normalization)  
> Step 3. Training  
> Step 4. Predict (first U-net and then watershed)  


## Discriptions
In folder `2DHeLa`, there are codes for training and predict in the 2-dimentional case.  
Training data comes from: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip (37MB)  
Testing data comes from: http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip (41MB)  

In folder `3DA549`, there are codes for training and predict in the 3-dimentional case.  
Training data comes from: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549.zip (243 MB)  
Testing data comes from: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C3DH-A549.zip (294 MB)

## Notes
In both folder, `calculate_acc.py` is useless, you could ignore it.  
You MUST change the PATH in `preprocessing.py`, `train.py`, `predict_dataset.py` and `create_tracking.py`.  

## How to run
1. Run `preprocessing.py` and then, training input images will be stored in /path/image and ground truths will be stored in /path/mask.  
2. Run `train.py`
3. Run `predict_dataset.py`
4. Run `create_tracking.py`
