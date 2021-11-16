
'''
===============================================================================
Define a set of parameters related to video segmentation
===============================================================================
'''

# Frequency of the annotation
ANNOTATION_INTERVAL = 100

# Frequency of the segmentation
# e.g., SEG_INTERVAL = 1, all frames are segmented
SEG_INTERVAL = 1

# Start Frame
# Start segmenting from this frame
STARTFRAME = 0

# Parameters for graph cut (do not change unless you know what you are doing)

# Number of foreground gaussian clusters
K_FGD = 8

# Number of background gaussian clusters
K_BGD = 8

GAMMA = 50

# Segmentation algorithm selection "BackFlow" or "MaskRCNN" or "OSVOS"
SEG_METHOD = "BackFlow"



