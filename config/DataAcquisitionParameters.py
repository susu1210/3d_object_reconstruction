"""
DataAcquisitionParameters.py
---------------

Define a set of parameters related to color and depth image acquisition

"""
# Camera serial number

# SERIAL = 'camera_image'
SERIAL = 'pybullet1-1'
# Depth camera intrinsic 
camera_intrinsics = {'616205001876': {'fx': 474.31524658203125, 'fy': 474.31524658203125,
                                      'height': 480, 'width': 640,
                                      'coeffs': [0.10955307632684708, 0.1833394467830658, 0.005065280944108963, 0.005705620162189007, -0.0680532231926918],
                                      'ppy': 244.9296875, 'ppx': 317.4243469238281, 'model': 2},
                     'pybullet1-1': {'fy': 415.6921938, 'fx': 415.6921938,
                                  'height': 480, 'width': 640,
                                  'ppy': 240, 'ppx': 320, 'model': 2},
                     'pybullet1-2': {'fx': 554.2562584, 'fy': 554.2562584,
                                  'height': 480, 'width': 640,
                                  'ppy': 240, 'ppx': 320, 'model': 2},
                     'pybullet2':{'fx': 866.0254038, 'fy': 866.0254038,
                                      'height': 480, 'width': 640,
                                      'ppy': 240, 'ppx': 320, 'model': 2},
                     'pybullet3-1':{'fx': 554.2562584, 'fy': 415.6921938,
                                      'height': 480, 'width': 640,
                                      'ppy': 240, 'ppx':320, 'model': 2},
                     'pybullet3-2':{'fx': 415.6921938, 'fy': 554.2562584,
                                      'height': 480, 'width': 640,
                                      'ppy': 240, 'ppx':320, 'model': 2},
                     'proj_view': {'height': 480, 'width': 640,
                                   'projection': (1.299038052558899, 0.0, 0.0, 0.0,
                                                  0.0, 1.7320507764816284, 0.0, 0.0,
                                                  0.0, 0.0, -1.0020020008087158, -1.0,
                                                  0.0, 0.0, -0.20020020008087158, 0.0),
                                   'view': (0.0, -0.24192193150520325, 0.970295786857605, 0.0,
                                            1.0, 0.0, -0.0, 0.0,
                                            0.0, 0.970295786857605, 0.24192193150520325, 0.0,
                                            -0.0, -0.2914799153804779, -1.1910423040390015, 1.0)},
                     'camera_image': {'height': 768,'width': 1024,
                                      'cameraUp': [0.0, 0.0, 1.0],
                                      }
                     }

#  this is the one used for ICRA demo
# camera_intrinsics = {'616205001876': {'fx': 618.1351928710938, 'fy': 618.13525390625, 'height': 480, 'width': 640, 'coeffs': [0.10955307632684708, 0.1833394467830658, 0.005065280944108963, 0.005705620162189007, -0.0680532231926918], 'ppy': 238.54409790039062, 'ppx': 312.1978759765625, 'model': 2}}




# Cut off depth reading greater than DEPTH_THRESH in meter 
DEPTH_THRESH = 0.5 
