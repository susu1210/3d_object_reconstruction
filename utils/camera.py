"""
camera.py
---------------

Functions related to data acquisition with cameras.

"""

import numpy as np
import math
import os
import pandas as pd
import pybullet as p
def convert_depth_frame_to_pointcloud(depth_image, serialnumber, camera_intrinsics):
        
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame : (m,n) uint16
		The depth_frame containing the depth map
        serialnumber : string
                The serial number of the camera used
	camera_intrinsics : dict 
                The intrinsic values of the depth imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	pointcloud : (m,n,3) float
		The corresponding pointcloud in meters

	"""
	# import cv2
	# cv2.imshow('depth', depth_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	[height, width] = depth_image.shape

	nx = np.linspace(0, width - 1, width)
	ny = np.linspace(0, height - 1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics[serialnumber]['ppx']) / camera_intrinsics[serialnumber]['fx']
	y = (v.flatten() - camera_intrinsics[serialnumber]['ppy']) / camera_intrinsics[serialnumber]['fy']
	depth_image = depth_image * 1.0 / 65535
	# print(depth_image[220][300:400])
	# depth_image = depth_image.astype(np.uint16)
	farVal = 100.0
	nearVal = 0.1
	depth_image = farVal * nearVal / (farVal - (farVal - nearVal) * depth_image)
	z = depth_image.flatten()
	x = np.multiply(x, z)
	y = np.multiply(y, z)
	pointcloud = np.dstack((x, y, z)).reshape((depth_image.shape[0]*depth_image.shape[1], 3))

	R = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, -math.pi * 104 / 180., 0]),
							 [0, 0, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2.]))[1]
	R = p.getMatrixFromQuaternion(R)
	R = np.asarray(R, dtype=np.float)
	R.shape = [3,3]
	T = [0.6 + 0.5 * math.cos(math.pi * 14 / 180.), 0, 0.45 + 0.5 * math.sin(math.pi * 14 / 180.)]
	T = np.asarray(T)
	pointcloud = np.transpose(np.dot(R, pointcloud.T)) + T
	pointcloud.shape=[depth_image.shape[0], depth_image.shape[1], 3]
	return pointcloud

def load_camera_intrinsic(path):
	camera_intrinsic = pd.read_csv(path + "/camera_intrinsic.csv")
	# print(camera_intrinsic.head())
	# print(camera_intrinsic.info())
	camForward = camera_intrinsic['camForward'][0]
	camForward = np.array(map(float, camForward[1:len(camForward) - 1].split(',')))

	horizon = camera_intrinsic['horizon'][0]
	horizon = np.array(map(float, horizon[1:len(horizon) - 1].split(',')))

	vertical = camera_intrinsic['vertical'][0]
	vertical = np.array(map(float, vertical[1: len(vertical) - 1].split(',')))

	dist = float(camera_intrinsic['dist'][0])

	camTarget = camera_intrinsic['camTarget'][0]
	camTarget = np.array(map(float, camTarget[1: len(camTarget) - 1].split(',')))

	fov = float(camera_intrinsic['fov'][0])
	farVal = float(camera_intrinsic['farVal'][0])
	nearVal = float(camera_intrinsic['nearVal'][0])

	viewMat = camera_intrinsic['viewMat'][0]
	viewMat = np.array(map(float, viewMat[1: len(viewMat) - 1].split(',')))
	viewMat = np.reshape(np.array(viewMat), [4, 4])

	projMat = camera_intrinsic['projMat'][0]
	projMat = np.array(map(float, projMat[1: len(projMat) - 1].split(',')))
	projMat = np.reshape(np.array(projMat), [4, 4])
	return camForward, horizon, vertical, dist, camTarget, fov, farVal, nearVal, viewMat, projMat

