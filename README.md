# 3D Object Reconstruction via RGB-D Video Segmentation

## Introduction

This is a repository for a 3D-Object-Reconstruction Project which processes the RGB-D video recoding all aspects of an object grasped by gripper. The code is consisted of 2 main parts, which can be used together or separately:

Segmentation:

Here two segmentation methods were provided, which can be chosen by setting the hyperparameter "SEG_METHOD" under /config/segmentationParameters.py"

1. An interactive video segmentation algorithm that requires user annotation at a specified interval and out result segmentation mask for all video frames. (See citation 1: https://motion.cs.illinois.edu/papers/ICRA2019-Wang-InHandScanning.pdf)

   ![BackFlow](doc/flowchart.png)

2. A single frame instance segmentation algorithm with MaskRCNN trained on several YCB object. (See citation 2: https://manipulation.csail.mit.edu/misc.html)

   

Reconstruction:

An local to global registration scheme that register the fragments into a global frame as colored .ply file. Here two reconstruction methods were provided, which can be chosen by setting the hyperparameter "RECON_METHOD" under /config/registrationParameters.py"

1. A reconstruction algorithm combined with SIFT feature matching and Iterative Closest Points(ICP).
2. A reconstruction algorithm directly using the sensor data from the robot's end effector( the final link observation of the gripper).


#### Citation
> @inproceedings{wang19,  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TITLE = {{In-hand Object Scanning via RGB-D Video Segmentation}},  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUTHOR = {Fan Wang and Kris Hauser},  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BOOKTITLE =  {ICRA},  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YEAR = {2019}  
> }
>
> Russ Tedrake. *Robot Manipulation: Perception, Planning, and Control (Course Notes for MIT 6.881).* Downloaded on 17/06/2012 from http://manipulation.csail.mit.edu/

### License

inHandObjectScanning is released under the MIT License (refer to the LICENSE file for details).


## Installation
### Installation --modified 
Installation has been tested on a fresh install of Ubuntu 20.04 with Python 3.8

Step 1:

Download the repository

```bash
git clone https://github.com/susu1210/3d_object_reconstruction.git
```

Step 2:

Upgrading any pre-install packages

```bash
sudo apt-get update
sudo apt-get upgrade
```
Step 3:

Install pip, a Python package manager, and update

```bash
sudo apt install python3-pip
```
Step 4:

Install the required packages through apt-get

```bash
sudo apt-get install build-essential cmake git pkg-config
```

Step 5:

Install the required packages through pip

```bash
sudo pip install numpy Cython scipy scikit-learn open3d scikit-image tqdm pykdtree opencv-python opencv-contrib-python pandas pymaxflow pymeshlab future
```

The modidified code with open3d version=0.12.0, pymaxflow version=1.2.13, pymeshlab version=0.2.1

Step 6:

Install pytorch for MaskRCNN

```bash
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Here is just a example for pytorch-cpu version 1.9.0. Details and more version via: https://pytorch.org/get-started/locally/

The Git repository include a trained MaskRCNN offered by MIT. More: https://manipulation.csail.mit.edu/segmentation.html




## Quick Start
1. Download the example dataset in [here](https://drive.google.com/open?id=1W0dkObpX7jzgENmzhVm7frhEwmkpOyKt) and extract it under the Data folder

2. Segment an object sequence
```python
python segmentation.py Data/juice
```

or you can provide "all" for path to segment all sequences in the Data folder
results will be saved in the "results" folder in under each sequence directory

you can also visualize the results saved as mp4 under sequence directory

(currently running the code will throw you a one time warning from slic, you can just ignore it)

3. Register segmented frames

```python
python registerwithkeyframes.py Data/juice
```

The constants related to segmentation and registration are defined in the respective files in the config folder. Please tune the knob accordingly (given that you know what you are doing)

The registered pointclouds are saved in sequencename.ply in the Data folder
To obtain watertight polygon mesh of the result, perform the following operations in meshLab:

I. compute normals
II. Screened Poisson surface reconstruction
III. (Optional) remove noise

![Meshlab](doc/MeshLab.jpeg) Check out (http://www.meshlab.net/) for installation and usage

## (Optional) Capture your own dataset

Save the color images in the cad folder as .jpg and depth image (aligned to color)
in the depth folder as uint16 png format.

Note that the depth images are interpolated within a 0-8 m range (in compliant with
the standard of intel realsense cameras), so that the 0 m reading has a pixel value
of 0, and a 8m reading has a pixel value of 65535.

e.g., To use a intel realsense camera for this step(ver > F200):

1. Install librealsense v1.12.1 and verify your installation by running examples.

2. Install the dependencies:

```bash
sudo pip install pycparser
```
3.  Install pyrealsense
'''bash
sudo pip install pyrealsense
'''
Note: pyrealsense changes its API quite often, if this happens again, you need to do some update on the record.py script.

4.  Plug in the camera and run!   

```python
python record.py <foldername>
```
e.g., to scan a random mug, run python record.py Data/Mug
The record script will automatically put data in the required folders, and generate empty folders for later steps.

The constants related to data acquisation are defineds in the config folder. Please tune the knob accordingly (given that you know what you are doing).

Enjoy the model building!

If you encounter any problems with the code, want to report bugs, etc. please contact me at fan.wang2[at]duke[dot]edu.





