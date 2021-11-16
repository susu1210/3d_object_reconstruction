'''
===============================================================================
Define a set of parameters related to fragment registration
===============================================================================
'''

import numpy as np

# Voxel size used to down sample the raw pointcloud for faster ICP
VOXEL_SIZE = 0.001


# Set up parameters for post-processing
# Voxel size for the complete mesh
VOXEL_R = 0.0002

# search for up to N frames for registration, odometry only N=1, all frames N = np.inf
# for any N!= np.inf, the refinement is local
K_NEIGHBORS = 10

# Specify an reconstruction algorithm
# "colored-icp", as in Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017 (slower)
# "point-to-plane", a coarse to fine implementation of point-to-plane icp (faster)
# "robot-joints", using robot sensor data to acquire rototation and translation of objects
RECON_METHOD = "robot-joints"


# set up the interval for keyframes

KEYFRAME_INTERVAL = 5

# Add surface_reconstruction_screened_poisson & remove_statistical_outlier or not
WATERTIGHT_POLYGON_MESH = True

