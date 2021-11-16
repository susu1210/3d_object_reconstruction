import math
import open3d as o3d
import numpy as np
import cv2
from utils.camera import *
from config.registrationParameters import *
from config.DataAcquisitionParameters import SERIAL,camera_intrinsics
from PIL import Image
import pymeshlab
import pybullet as p

def load_colordepth(path, Filename):
    # mask_path = path + 'results/%s.png' % (Filename)
    # ignoring loss of segmented masks and ground-truth
    mask_path = path + 'annotations/%s.png' % (Filename)
    exist = os.path.isfile(mask_path)
    if exist:
        mask = cv2.imread(mask_path, 0)
    else:
        mask_path = path + 'annotations/%s.png' % (Filename)
    mask = cv2.imread(mask_path, 0)
    img_file = path + 'color/%s.jpg' % (Filename)
    cad = cv2.imread(img_file)
    cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)
    segmented = cv2.bitwise_and(cad, cad, mask=mask)
    # cv2.imshow('rgb',segmented)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    depth_file = path + 'depth/%s.png' % (Filename)
    depth = Image.open(depth_file)
    depth = np.array(depth, dtype=np.uint16)

    pointcloud = convert_depth_frame_to_pointcloud(depth, SERIAL, camera_intrinsics)
    # pointcloud = creat_pointcloud_from_camera_image(depth, SERIAL, camera_intrinsics)
    # pointcloud = get_pointcloud_from_camera_image(path, depth, SERIAL, camera_intrinsics)

    return (cad, segmented, depth, pointcloud)

def surface_reconstruction_screened_poisson(input_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    ms.compute_normals_for_point_sets()
    ms.surface_reconstruction_screened_poisson()
    cm = ms.current_mesh()
    print(cm)
    output_path=input_path.split('.')[0]+"_filtered.ply"
    ms.save_current_mesh(output_path)
    filtered_mesh = o3d.io.read_point_cloud(output_path)
    return filtered_mesh
if __name__ == "__main__":
    YcbObjects = ["YcbMustardBottle",
                  "YcbGelatinBox",
                  "YcbPottedMeatCan",
                  "YcbTomatoSoupCan",
                  "YcbCrackerBox",
                  "YcbSugarBox",
                  "YcbBanana",
                  "YcbTennisBall"]
    DataPath = ["Data", "Data_new"]
    SegMethods = ["BackFlow", "OSVOS"]
    ReconMethods = ["point-to-plane", "robot-joints"]
    for obj in YcbObjects:
        for d in DataPath:
            for s in SegMethods:
                for r in ReconMethods:
                    print(d, obj, s, r)
                    path = os.path.join(os.getcwd(), d, obj)
                    pcd = o3d.io.read_point_cloud(os.path.join(path, obj+'_'+s+'_'+r+'.ply'))
                    o3d.visualization.draw_geometries([pcd])
