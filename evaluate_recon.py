import os.path
import pandas as pd
import open3d as o3d
import numpy as np
import pybullet as p
import xml.dom.minidom as xmldom
import pymeshlab

def parse_xml(fn):
    rpy = {}
    xyz = {}
    xml_file = xmldom.parse(fn)
    rootNode = xml_file.documentElement
    print(rootNode.nodeName)
    links = rootNode.getElementsByTagName("link")
    for link in links:
        if link.hasAttribute("name") and (link.getAttribute("name")=="baseLink" or link.getAttribute("name")=="baselink"):
            inertial = link.getElementsByTagName("inertial")[0]
            visual = link.getElementsByTagName("visual")[0]
            if len(inertial.getElementsByTagName("origin"))>0 :
                origin = inertial.getElementsByTagName("origin")[0]
                temp = list(map(float, origin.getAttribute("rpy").split(' ')))
                rpy['inertial']  = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(temp))).reshape([3, 3])
                temp = list(map(float, origin.getAttribute("xyz").split(' ')))
                xyz['inertial'] = np.asarray(temp)
            if len(visual.getElementsByTagName("origin")) > 0:
                origin = visual.getElementsByTagName("origin")[0]
                temp = list(map(float, origin.getAttribute("rpy").split(' ')))
                rpy['visual']  = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(temp))).reshape([3, 3])
                temp = list(map(float, origin.getAttribute("xyz").split(' ')))
                xyz['visual'] = np.asarray(temp)
            break
    print(rpy, xyz)
    return rpy, xyz


def compute_unsigned_distance_and_closest_goemetry(scene, query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    return distance, closest_points['geometry_ids'].numpy()

def compute_signed_distance_and_closest_goemetry(scene, query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points['geometry_ids'].numpy()

YcbObjects = ["YcbMustardBottle",
              "YcbGelatinBox",
              "YcbPottedMeatCan",
              "YcbTomatoSoupCan",
              "YcbCrackerBox",
              "YcbSugarBox",
              "YcbBanana",
              "YcbTennisBall"]
DataPath = ["Data", "Data_stuck"]
SegMethods = ["BackFlow", "OSVOS"]
ReconMethods = ["point-to-plane", "robot-joints"]

if __name__ == '__main__':
    mean_usdf = []
    mean_sdf = []
    mean_squared_sdf = []
    mean_hausdorff_distance = []
    min_hausdorff_distance = []
    max_hausdorff_distance = []
    RMS_hausdorff_distance =[]
    obj_record = []
    data_record = []
    seg_record = []
    recon_record = []

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    for obj in YcbObjects:
        if obj == "YcbSugarBox":
            mesh_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),
                                     "pybullet-object-models", "pybullet_object_models",
                                     "ycb_objects", obj, "textured.obj")
        elif obj == "YcbTennisBall":
            mesh_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),
                                     "pybullet-object-models", "pybullet_object_models",
                                     "ycb_objects", obj, "textured_reoriented.obj")
        else:
            mesh_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),
                                "pybullet-object-models", "pybullet_object_models",
                                "ycb_objects", obj, "textured_simple_reoriented.obj")
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),
                            "pybullet-object-models", "pybullet_object_models",
                            "ycb_objects", obj, "model.urdf")
        rpy, xyz = parse_xml(urdf_path)

        mesh = o3d.io.read_triangle_mesh(mesh_path, True)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_path)

        scene = o3d.t.geometry.RaycastingScene()
        mesh_cpu = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_cpu)

        count = 0
        for d in DataPath:
            for s in SegMethods:
                for r in ReconMethods:
                    count += 1
                    print(d, obj, s, r)
                    path = os.path.join(os.getcwd(), d, obj)
                    pcd = o3d.io.read_point_cloud(os.path.join(path, obj + '_' + s + '_' + r + '.ply'))
                    pcd1 = o3d.io.read_point_cloud(os.path.join(path, obj + '_' + s + '_' + r + '.ply'))
                    robot_joints = pd.read_csv(path + "/robot_joints.csv", index_col='filenames')
                    ObjectR = robot_joints['ObjectRotationMatrices'][1]
                    ObjectPositions = robot_joints['ObjectPositions'][1]
                    ObjectOrientation = list(map(float, ObjectR[1:len(ObjectR) - 1].split(',')))
                    ObjectOrientation = np.reshape(np.array(ObjectOrientation), [3, 3])
                    ObjectPositions = np.array( list(map(float, ObjectPositions[1:len(ObjectPositions) - 1].split(','))))
                    # o3d.visualization.draw_geometries([mesh,pcd1,coordinate])
                    points = ( np.dot(np.linalg.inv(ObjectOrientation), (np.asarray(pcd.points) - ObjectPositions).T) ).T
                    if 'inertial' in rpy:
                        points = ( np.dot( rpy['inertial'],  points.T)).T + xyz['inertial']
                    if 'visual' in rpy:
                        points = (np.dot(np.linalg.inv(rpy['visual']), (points- xyz['visual']).T )).T
                    pcd.points = o3d.utility.Vector3dVector(points)
                    o3d.io.write_point_cloud(os.path.join(path, obj + '_' + s + '_' + r + '_origin.ply'), pcd)
                    ms.load_new_mesh(os.path.join(path, obj + '_' + s + '_' + r + '_origin.ply'))
                    temp_haus_dis = ms.hausdorff_distance(sampledmesh=0, targetmesh=count, samplenum=8000)
                    ms.set_current_mesh(new_curr_id=count)
                    ms.delete_current_mesh()
                    mean_hausdorff_distance.append(temp_haus_dis['mean'])
                    min_hausdorff_distance.append(temp_haus_dis['min'])
                    max_hausdorff_distance.append(temp_haus_dis['max'])
                    RMS_hausdorff_distance.append(temp_haus_dis['RMS'])
                    print("mean hausdorff distance ", mean_hausdorff_distance[-1])
                    print("rms hausdorff distance ", RMS_hausdorff_distance[-1])
                    # points = pcd.points + np.asarray([0.2,0,0])
                    # pcd1.points = o3d.utility.Vector3dVector(points)
                    # o3d.visualization.draw_geometries([mesh,pcd1,coordinate])

                    # compute range
                    usdf, _ = compute_unsigned_distance_and_closest_goemetry(scene, query_points = np.asarray(pcd.points).astype(np.float32))
                    sdf, _ = compute_signed_distance_and_closest_goemetry(scene, query_points = np.asarray(pcd.points).astype(np.float32))
                    temp_usdf = np.mean(usdf)
                    temp_sdf = np.mean(sdf)
                    temp_ssdf = np.mean(np.square(sdf))
                    print("mean usdf ", temp_usdf)
                    print("mean sdf ", temp_sdf)
                    print('mean squared sdf', temp_ssdf)
                    mean_usdf.append(temp_usdf)
                    mean_sdf.append(temp_sdf)
                    mean_squared_sdf.append(temp_ssdf)
                    obj_record.append(obj)
                    data_record.append(d)
                    seg_record.append(s)
                    recon_record.append(r)

    result_df = pd.DataFrame({'Data Path': data_record,
                              'Object Name': obj_record,
                              'Segmentation Method': seg_record,
                              'Reconstruction Method': recon_record,
                              'mean usdf': mean_usdf,
                              'mean sdf': mean_sdf,
                              'mean squared sdf': mean_squared_sdf,
                              'mean haus dist': mean_hausdorff_distance,
                              'min haus dist': min_hausdorff_distance,
                              'max haus dist': max_hausdorff_distance,
                              'RMS haus dist': RMS_hausdorff_distance
                              })
    result_df.to_csv(os.path.join(os.getcwd(), "recon_comparison.csv"), index=False)
