"""
registerSegments.py
---------------

Main Function for registering (aligning) colored point clouds with ICP/feature 
matching as well as pose graph optimizating

"""
# import png
from PIL import Image
import csv
import open3d as o3d
import pymeshlab
import numpy as np
import cv2
import os
import glob
from utils.ply import Ply
from utils.camera import *
from registration import icp, feature_registration, match_ransac, rigid_transform_3D
from tqdm import trange
from pykdtree.kdtree import KDTree
import time
import sys
from config.registrationParameters import *
from config.segmentationParameters import SEG_INTERVAL, STARTFRAME, SEG_METHOD,ANNOTATION_INTERVAL
from config.DataAcquisitionParameters import SERIAL,camera_intrinsics
import pandas as pd

# Set up parameters for registration
# voxel sizes use to down sample raw pointcloud for fast ICP
voxel_size = VOXEL_SIZE
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

# Set up parameters for post-processing
# Voxel size for the complete mesh
voxel_Radius = VOXEL_R

# Point considered an outlier if more than inlier_Radius away from other points  
inlier_Radius = voxel_Radius * 2.5

# search for up to N frames for registration, odometry only N=1, all frames N = np.inf
N_Neighbours = K_NEIGHBORS


def post_process(originals, voxel_Radius, inlier_Radius):
     """
    Merge segments so that new points will not be add to the merged
    model if within voxel_Radius to the existing points, and keep a vote
    for if the point is issolated outside the radius of inlier_Radius at 
    the timeof the merge

    Parameters
    ----------
    originals : List of open3d.Pointcloud classe
      6D pontcloud of the segments transformed into the world frame
    voxel_Radius : float
      Reject duplicate point if the new point lies within the voxel radius
      of the existing point
    inlier_Radius : float
      Point considered an outlier if more than inlier_Radius away from any 
      other points

    Returns
    ----------
    points : (n,3) float
      The (x,y,z) of the processed and filtered pointcloud
    colors : (n,3) float
      The (r,g,b) color information corresponding to the points
    vote : (n, ) int
      The number of vote (seen duplicate points within the voxel_radius) each 
      processed point has reveived
    """

     for point_id in trange(len(originals)):

          if point_id == 0:
               vote = np.zeros(len(originals[point_id].points))
               points = np.array(originals[point_id].points,dtype = np.float64)
               colors = np.array(originals[point_id].colors,dtype = np.float64)

          else:
       
               points_temp = np.array(originals[point_id].points,dtype = np.float64)
               colors_temp = np.array(originals[point_id].colors,dtype = np.float64)
               
               dist , index = nearest_neighbour(points_temp, points)
               new_points = np.where(dist > voxel_Radius)
               points_temp = points_temp[new_points]
               colors_temp = colors_temp[new_points]
               inliers = np.where(dist < inlier_Radius)
               vote[(index[inliers],)] += 1
               vote = np.concatenate([vote, np.zeros(len(points_temp))])
               points = np.concatenate([points, points_temp])
               colors = np.concatenate([colors, colors_temp])

     return (points,colors,vote) 

def surface_reconstruction_screened_poisson(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.compute_normals_for_point_sets()
    ms.surface_reconstruction_screened_poisson()
    ms.save_current_mesh(path)
    filtered_mesh = o3d.io.read_point_cloud(path)
    return filtered_mesh

def full_registration(pcds_down,cads,depths, max_correspondence_distance_coarse,max_correspondence_distance_fine):

    """
    perform pairwise registration and build pose graph for up to N_Neighbours

    Parameters
    ----------
    pcds_down : List of open3d.Pointcloud instances
      Downampled 6D pontcloud of the unalligned segments
    max_correspondence_distance_coarse : float
      The max correspondence distance used for the course ICP during the process
      of coarse to fine registration
    max_correspondence_distance_fine : float
      The max correspondence distance used for the fine ICP during the process 
      of coarse to fine registration

    Returns
    ----------
    pose_graph: an open3d.PoseGraph instance
       Stores poses of each segment in the node and pairwise correlation in vertice
    """
    
    global N_Neighbours
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds_down)

    for source_id in trange(n_pcds):
        for target_id in range(source_id + 1, min(source_id + N_Neighbours,n_pcds)):

            # derive pairwise registration through feature matching
            color_src = cads[source_id]
            depth_src = depths[source_id]
            color_dst = cads[target_id]
            depth_dst = depths[target_id]

            res = feature_registration((color_src, depth_src),
                                       (color_dst, depth_dst))

            if res is None:
                # if feature matching fails, perform pointcloud matching
                transformation_icp, information_icp = icp(
                        pcds_down[source_id], pcds_down[target_id],max_correspondence_distance_coarse,
                        max_correspondence_distance_fine, method = RECON_METHOD)

            else:
                transformation_icp = res
                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    pcds_down[source_id], pcds_down[target_id], max_correspondence_distance_fine,
                    transformation_icp)

            information_icp *= 1.2 ** (target_id - source_id - 1)
            if target_id == source_id + 1:
                # odometry
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                      transformation_icp, information_icp, uncertain=False))
            else:
                # loop closure
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                      transformation_icp, information_icp, uncertain=True))

    return pose_graph


def joints_full_registration(pcds_down, LinkOrientations, LinkPositions):
    """
    perform pairwise registration using robot end-effector poses and build pose graph

    Parameters
    ----------
    pcds_down : List of open3d.Pointcloud instances
      Downampled 6D pontcloud of the unalligned segments
    LinkOrientations : List of end-effector Orientations
    LinkPositions : List of end-effector positions

    Returns
    ----------
    pose_graph: an open3d.PoseGraph instance
       Stores poses of each segment in the node and pairwise correlation in vertice
    """

    global N_Neighbours
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds_down)

    for source_id in trange(n_pcds):
        for target_id in range(source_id + 1, min(source_id + N_Neighbours, n_pcds)):

            R1 , R2 = LinkOrientations[source_id] , LinkOrientations[target_id]
            T1 , T2 = LinkPositions[source_id] , LinkPositions[target_id]
            transformation_icp = calculate_transformation(R1 , R2, T1, T2)
            information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds_down[source_id], pcds_down[target_id], max_correspondence_distance_fine,
                transformation_icp)


            if target_id == source_id + 1:
                # odometry
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                      transformation_icp, information_icp, uncertain=False))
            else:
                # loop closure
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                      transformation_icp, information_icp, uncertain=True))
    return pose_graph

def calculate_transformation(R1, R2, T1, T2):
    R = np.dot(R2, np.linalg.inv(R1))
    T = T2 - np.dot(T1, np.dot(np.linalg.inv(R1).T, R2.T))

    transformation_icp = [[R[0][0],R[0][1],R[0][2],T[0]],
                 [R[1][0],R[1][1],R[1][2],T[1]],
                 [R[2][0],R[2][1],R[2][2],T[2]],
                 [0,0,0,1]]

    return transformation_icp

def load_robot_joints(path, keyframe_ids):
    robot_joints = pd.read_csv(path+"/robot_joints.csv", index_col='filenames')
    LinkR = robot_joints['LinkRotationMatrices']
    LinkPositions = robot_joints['LinkPositions']
    Rs=[]
    Ts=[]
    for filename in keyframe_ids:
        R = list(map(float, LinkR[filename][1:len(LinkR[filename]) - 1].split(',')))
        R = np.reshape(np.array(R), [3, 3])
        T = np.array(list(map(float, LinkPositions[filename][1:len(LinkPositions[filename]) - 1].split(','))))
        Rs.append(R)
        Ts.append(T)
    return Rs, Ts

def load_object_states(path, keyframe_ids):
    robot_joints = pd.read_csv(path+"/robot_joints.csv", index_col='filenames')
    ObjectR = robot_joints['ObjectRotationMatrices']
    ObjectPositions = robot_joints['ObjectPositions']
    Rs=[]
    Ts=[]
    for filename in keyframe_ids:
        R = list(map(float, ObjectR[filename][1:len(ObjectR[filename]) - 1].split(',')))
        R= np.reshape(np.array(R), [3, 3])
        T = np.array(list(map(float, ObjectPositions[filename][1:len(ObjectPositions[filename]) - 1].split(','))))
        Rs.append(R)
        Ts.append(T)
    return Rs, Ts

def load_colordepth(path,Filename):

     mask_path = path + SEG_METHOD+'_results/%s.png' % (Filename)
     # ignoring loss of segmented masks and ground-truth
     # mask_path = path + 'annotations/%s.png' % (Filename)
     exist = os.path.isfile(mask_path)
     if exist:
         mask = cv2.imread(mask_path, 0)
     else:
         mask_path = path + 'annotations/%s.png' % (Filename)
         mask = cv2.imread(mask_path, 0)

     img_file = path + 'cad/%s.jpg' % (Filename)
     cad = cv2.imread(img_file)
     cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)
     # cv2.imshow("cad", cad)
     # cv2.imshow("mask", mask)
     # cv2.waitKey(0)
     # cv2.destroyAllWindows()
     segmented = cv2.bitwise_and(cad,cad,mask = mask)
     # if Filename<=1000:
     #     bbx = np.zeros_like(mask,dtype='uint8')
     #     bbx[180:300,235:400] = 1
     #     segmented = cv2.bitwise_and(segmented, segmented, mask=bbx)
     # cv2.imshow("segmented", segmented)
     # cv2.waitKey(0)
     # cv2.destroyAllWindows()


     # color_thr = 125
     # color_mask = np.asarray(np.logical_and((segmented[:,:,0]>=color_thr),
     #                                        np.logical_and( (segmented[:,:,1]>=color_thr),(segmented[:,:,2]>=color_thr))),dtype=np.uint8)*255
     # color_mask = np.asarray(np.logical_xor(mask, color_mask), dtype=np.uint8)*255
     # segmented = cv2.bitwise_and(segmented, segmented, mask=color_mask)

     # cv2.imshow("segmented", segmented)
     # cv2.waitKey(0)
     # cv2.destroyAllWindows()
     depth_file = path + 'depth/%s.png' % (Filename)
     depth = Image.open(depth_file)
     depth = np.array(depth, dtype=np.uint16)

     if SEG_METHOD == "MaskRCNN":
         depth_thr = 0.9 * 65535
         depth_mask = np.asarray(depth>= depth_thr, dtype=np.uint8) * 255
         depth_mask = np.asarray(np.logical_xor(mask, depth_mask), dtype=np.uint8) * 255
         segmented = cv2.bitwise_and(segmented, segmented, mask=depth_mask)

     # cv2.imshow("segmented", segmented)
     # cv2.waitKey(0)
     # cv2.destroyAllWindows()

     pointcloud = convert_depth_frame_to_pointcloud(depth, SERIAL, camera_intrinsics)

     return (segmented, depth, pointcloud)

def select_keyframes(path):
     
    keyframe_ids = []

    Filename = SEG_INTERVAL + STARTFRAME
    Matched = True
    try:

        while True:


           if Matched:
                interval = int(KEYFRAME_INTERVAL/SEG_INTERVAL)*SEG_INTERVAL
           else:
                interval -= SEG_INTERVAL



           color_src, mask_src, pointcloud_src = load_colordepth(path,Filename)
           color_dst, mask_dst, pointcloud_dst = load_colordepth(path,Filename + interval)

           res = feature_registration((color_src, pointcloud_src),
                                      (color_dst, pointcloud_dst))
           if res is not None:
                if Filename not in keyframe_ids:
                     keyframe_ids.append(Filename)
                keyframe_ids.append(Filename+interval)
                Filename += interval
                Matched = True
                continue
           else:
                segmented_mask_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2GRAY)
                segmented_mask_dst = cv2.cvtColor(color_dst, cv2.COLOR_BGR2GRAY)
                segmented_mask_src[mask_src==0] = 0
                segmented_mask_dst[mask_dst==0] = 0

                src_show = pointcloud_src[segmented_mask_src>0]
                dst_show = pointcloud_dst[segmented_mask_dst>0]

                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(src_show)
                source.colors = o3d.utility.Vector3dVector(color_src[segmented_mask_src>0])

                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(dst_show)
                target.colors = o3d.utility.Vector3dVector(color_dst[segmented_mask_dst>0])


                voxel_radius = [ 0.001, 0.001, 0.001 ]
                max_iter = [ 80, 80, 80 ]
                current_transformation = np.identity(4)

                for scale in range(3):
                     iter = max_iter[scale]
                     r = voxel_radius[scale]

                     source_down = source.voxel_down_sample(voxel_size = r)
                     target_down = target.voxel_down_sample(voxel_size = r)


                     source_down.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = r * 2, max_nn = 30))
                     target_down.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = r * 2, max_nn = 30))

                     result_icp = o3d.pipelines.registration.registration_colored_icp(target_down, source_down,
                                    r, current_transformation,
                                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                                    o3d.pipelines.registration.ICPConvergenceCriteria(
                                    relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=iter))
                     current_transformation = result_icp.transformation

                if len(np.asarray(result_icp.correspondence_set))> 1000:

                     Matched = True
                     if Filename not in keyframe_ids:
                          keyframe_ids.append(Filename)
                          keyframe_ids.append(Filename+interval)
                     Filename += interval
                     continue

                elif interval - SEG_INTERVAL < SEG_INTERVAL:

                     Matched = True
                     if Filename not in keyframe_ids:
                          keyframe_ids.append(Filename)
                          keyframe_ids.append(Filename+interval)
                     Filename += interval
                     continue
                else:
                     Matched = False
    except:
         return keyframe_ids


def load_sources(path, keyframe_ids):
     
    """
    load segmented color images, depth readings, pointcloud(unsampled) and
    pointcloud(downsampled) for at the specified keyframe intervals
    """
    
    global voxel_size
    cads = []
    depths = []
    pcds= []
    pcds_down = []

    num_frames = len(glob.glob1(path + "cad","*.jpg"))
    for Filename in keyframe_ids:
        segmented, mask, depth = load_colordepth(path,Filename)
        cads.append(segmented)
        depths.append(depth)

        color_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        color_gray[mask == 0] = 0
        mask = color_gray

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(depth[mask>0])
        source.colors = o3d.utility.Vector3dVector(segmented[mask>0])
        # o3d.visualization.draw_geometries([source])
        pcds.append(source)
        pcd_down = source.voxel_down_sample(voxel_size = voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = 0.002 * 2, max_nn = 30))
        pcds_down.append(pcd_down)

    return (cads,depths,pcds,pcds_down)

     



def nearest_neighbour(a, b):
    """
    find the nearest neighbours of a in b using KDTree
    Parameters
    ----------
    a : (n, ) numpy.ndarray
    b : (n, ) numpy.ndarray

    Returns
    ----------
    dist : n float
      Euclidian distance of the closest neighbour in b to a
    index : n float
      The index of the closest neighbour in b to a in terms of Euclidian distance
    """
    tree = KDTree(b)
    dist, index = tree.query(a)
    return (dist, index)


def print_usage():

    print( "Usage: registerwithkeyframes.py <path>")
    print( "path: [data_path]/all or name of the folder")
    print( "e.g., registerwithkeyframes.py Data/all, registerwithkeyframes.py Data_new/all, "
           "registerwithkeyframes.py Data/Cheezit")
    
    
if __name__ == "__main__":
  
    try:
        if sys.argv[1][-3:] == "all":
            folders = glob.glob(sys.argv[1][:-3]+"*/")
        elif sys.argv[1]+"/" in glob.glob("Data_new/*/") \
                or sys.argv[1]+"/" in glob.glob("Data/*/")\
                or sys.argv[1]+"/" in glob.glob("Data_stuck/*/"):
            folders = [sys.argv[1]+"/"]
        else:
            print_usage()
            exit()
    except:
        print_usage()
        exit()
    print("SEG_METHOD ", SEG_METHOD)
    print("RECON_METHOD ",RECON_METHOD)

    for path in folders:
        print(path)
        print("select key frames ...")
        exist = os.path.isfile(path + "keyframe_ids.csv")
        if exist:
            with open(path + "keyframe_ids.csv", 'r') as f:
                reader = csv.reader(f)
                keyframe_ids = [row[0] for row in reader]
                keyframe_ids = list(map(int, keyframe_ids[1:]))
        else:
            keyframe_ids = select_keyframes(path)
            key = pd.DataFrame({'keyframe_ids': np.array(keyframe_ids, dtype=int)})
            key.to_csv(path + 'keyframe_ids.csv', index=False)
        # keyframe_ids = list(range(1,1205,5))

        print("Load pointclouds ...")
        cads,depths,originals,pcds_down = load_sources(path, keyframe_ids)
        print("Full registration ...")
        if RECON_METHOD == "robot-joints":
            LinkOrientations, LinkPositions = load_robot_joints(path, keyframe_ids)
            ObjectOrientations, ObjectPositions = load_object_states(path, keyframe_ids)
            pose_graph = joints_full_registration(pcds_down, LinkOrientations, LinkPositions)
            # pose_graph = joints_full_registration(pcds_down, ObjectOrientations, ObjectPositions)
        else:
            pose_graph = full_registration(pcds_down, cads, depths,
                                           max_correspondence_distance_coarse,
                                           max_correspondence_distance_fine)

        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance = max_correspondence_distance_fine,
                edge_prune_threshold = 0.25,
                reference_node = 0)
        o3d.pipelines.registration.global_optimization(pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)


        print( "Merge segments")

        for point_id in trange(len(originals)):

            originals[point_id].transform(pose_graph.nodes[point_id].pose)

        print("Apply post processing")
        points, colors, vote = post_process(originals, voxel_Radius, inlier_Radius)
        ply = Ply(points[vote>1], colors[vote>1])

        pg = pd.DataFrame({'PoseGraph': pose_graph}, index=[0])
        pg.to_csv(os.path.join(path, "pose_graph.csv"))

        meshfile = '%s.ply' % (path[:-1]+'_'+SEG_METHOD+'_'+RECON_METHOD)

        ply.write(meshfile)
        print("Mesh saved")

        # Surface_reconstruction_screened_poisson & remove_statistical_outlier
        # Surface_reconstruction_screened_poisson & remove_statistical_outlier
        if WATERTIGHT_POLYGON_MESH:
            pcd = o3d.io.read_point_cloud(meshfile)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30,
                                                      std_ratio=2.0)
            output_path = os.path.join(os.getcwd(),meshfile.split('.')[0] + "_filtered.ply")
            o3d.io.write_point_cloud(output_path, cl)
            pcd = surface_reconstruction_screened_poisson(output_path)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coordinate])
        else:
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([o3d.io.read_point_cloud(meshfile),coordinate])
            pass
