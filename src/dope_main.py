#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function
import yaml
import sys 

import numpy as np
import cv2

import rospy
import rospkg
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor_msg
from geometry_msgs.msg import PoseStamped
from dope.msg import AdvYCBFrame as Adv_msg

from PIL import Image
from PIL import ImageDraw

import os
import tf
import transforms3d as tf3

# Import DOPE code
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from cuboid import *
from detector import *

### Global Variables
g_bridge = CvBridge()
g_img = None
g_draw = None
g_scene = None


### Basic functions
def __image_callback(msg):
    '''Image callback'''
    global g_img
    global g_scene
    g_img = g_bridge.imgmsg_to_cv2(msg.image, "rgb8")
    g_scene = msg.scene_name
    # cv2.imwrite('img.png', cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging
    # print(g_scene)


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0]-pointRadius, 
            point[1]-pointRadius, 
            point[0]+pointRadius, 
            point[1]+pointRadius
        ]
        g_draw.ellipse(xy, 
            fill=pointColor, 
            outline=pointColor
        )

def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across 
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)
    
    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)
    
    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius = 4)
    DrawDot(points[1], pointColor=color, pointRadius = 4)

    # draw x on the top 
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)

def writeMat2File(fo, mat):
    for line in np.matrix(mat):
        np.savetxt(fo, line, fmt='%f')

def run_dope_node(params, freq=5):
    '''Starts ROS node to listen to image topic, run DOPE, and publish DOPE results'''

    global g_img
    global g_draw
    global g_scene

    pubs = {}
    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] =\
            ModelData(
                model, 
                g_path2package + "/weights/" + params['weights'][model]
            )
        models[model].load_net_model()
        
        draw_colors[model] = \
            tuple(params["draw_colors"][model])
        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )
        pubs[model] = \
            rospy.Publisher(
                '{}/pose_{}'.format(params['topic_publishing'], model), 
                PoseStamped, 
                queue_size=10
            )
        pub_dimension[model] = \
            rospy.Publisher(
                '{}/dimension_{}'.format(params['topic_publishing'], model),
                String, 
                queue_size=10
            )

    # Start ROS publisher
    pub_rgb_dope_points = \
        rospy.Publisher(
            params['topic_publishing']+"/rgb_points", 
            ImageSensor_msg, 
            queue_size=10
        )
    
    # Starts ROS listener
    rospy.Subscriber(
        topic_cam, 
        Adv_msg, 
        __image_callback
    )

    # Initialize ROS node
    rospy.init_node('dope_vis', anonymous=True)
    rate = rospy.Rate(freq)

    print ("Running DOPE...  (Listening to camera topic: '{}')".format(topic_cam)) 
    print ("Ctrl-C to stop")

    if os.path.isdir('./benchmark_out'):
        os.system('rm -rf ./benchmark_out')

    transMat_ycbObj2dopeObj = {}
    Z = [1.0, 1.0, 1.0]

    T = [0.0, -0.00213, -0.00554]
    R = tf3.quaternions.quat2mat([0.5, -0.5, 0.5, -0.5]) # 003_cracker_box
    transMat_ycbObj2dopeObj['003_cracker_box'] = np.linalg.inv(tf3.affines.compose(T, R, Z))

    T = [0.00197, -0.00793, 0.00071]
    R = tf3.quaternions.quat2mat([0.707, 0.707, 0.017, -0.017]) # 010_potted_meat_can
    transMat_ycbObj2dopeObj['010_potted_meat_can'] = np.linalg.inv(tf3.affines.compose(T, R, Z))

    T = [0.00035, -0.00831, -0.00142]
    R = tf3.quaternions.quat2mat([0.707, 0.707, -0.0, 0.0]) # 005_tomato_soup_can
    transMat_ycbObj2dopeObj['005_tomato_soup_can'] = np.linalg.inv(tf3.affines.compose(T, R, Z))

    T = [0.0, 0.0113, 0.00247]
    R = tf3.quaternions.quat2mat([0.693, 0.693, -0.142, 0.142]) # 006_mustard_bottle
    transMat_ycbObj2dopeObj['006_mustard_bottle'] = np.linalg.inv(tf3.affines.compose(T, R, Z))

    while not rospy.is_shutdown():
        if g_img is not None and g_scene is not None:
            # Copy and draw image
            img_copy = g_img.copy()
            im = Image.fromarray(img_copy)
            g_draw = ImageDraw.Draw(im)

            fn_PoseResult = os.path.join('./benchmark_out', g_scene, 'times1/pose_result.txt')
            if not os.path.isdir(os.path.dirname(fn_PoseResult)):
                os.makedirs(os.path.dirname(fn_PoseResult))
            if os.path.exists(fn_PoseResult):
                os.remove(fn_PoseResult)

            for m in models:

                fn_debug = os.path.join('./benchmark_out', g_scene, 'times1/' + m + '_debug.txt')
                
                # Detect object
                results = ObjectDetector.detect_object_in_image(
                            models[m].net, 
                            pnp_solvers[m],
                            g_img,
                            config_detect
                            )
                save_x = 0.0
                save_y = 0.0
                save_z = 0.0
                save_rx = 0.0
                save_ry = 0.0
                save_rz = 0.0
                save_rw = 0.0
                # Publish pose and overlay cube on image
                # for i_r, result in enumerate(results):
                print('\n\ng_scene: ' + g_scene)
                print('m: ' + m)

                save_i_r = -1
                for i, item in enumerate(results):
                    if item["location"] is not None:
                        save_i_r = i
                        break

                with open(fn_debug, 'w') as f_debug:
                    if save_i_r >= 0:

                        result = results[save_i_r]
                        assert(result["location"] is not None)

                        loc = result["location"]
                        ori = result["quaternion"]                    
                        msg = PoseStamped()
                        msg.header.frame_id = params["frame_id"]
                        msg.header.stamp = rospy.Time.now()
                        CM2M = 100
                        msg.pose.position.x = loc[0] / CM2M
                        msg.pose.position.y = loc[1] / CM2M
                        msg.pose.position.z = loc[2] / CM2M
                        msg.pose.orientation.x = ori[0]
                        msg.pose.orientation.y = ori[1]
                        msg.pose.orientation.z = ori[2]
                        msg.pose.orientation.w = ori[3]

                        f_debug.write('Raw output - Camera view pose [x, y, z, qx, qy, qz, qw]:\n')
                        f_debug.write('{},{},{},{},{},{},{}\n\n'.format(loc[0], loc[1], loc[2], ori[0], ori[1], ori[2], ori[3]))

                        # convert to XYZ RPY (rot is ZYX)
                        T = [loc[0] / CM2M, loc[1] / CM2M, loc[2] / CM2M]
                        R = tf3.quaternions.quat2mat([ori[3], ori[0], ori[1], ori[2]])
                        Z = [1.0, 1.0, 1.0]
                        poseMat = tf3.affines.compose(T, R, Z)

                        f_debug.write('PoseMat (camera):\n')
                        writeMat2File(f_debug, poseMat)
                        f_debug.write('\n')

                        # print("Camera frame pose")
                        # print(poseMat)

                        # transform to baselink
                        T = [0.00762554, 0.793312, 0.991729]
                        R = tf3.quaternions.quat2mat([-0.188503, -0.678222, 0.681912, -0.19869])
                        Z = [1.0, 1.0, 1.0]
                        transMat_base2camera = tf3.affines.compose(T, R, Z)
                        transMat_camera2base = np.linalg.inv(transMat_base2camera)
                        # transMat_camera2base = transMat_base2camera

                        f_debug.write('transMat base2camera (coordinate trans):\n')
                        writeMat2File(f_debug, transMat_base2camera)
                        f_debug.write('\n')
                        f_debug.write('transMat camera2base (coordinate trans):\n')
                        writeMat2File(f_debug, transMat_camera2base)
                        f_debug.write('\n')

                        # print("transMat_camera2base")
                        # print(transMat_camera2base)

                        poseMat = np.matmul(transMat_camera2base, np.matmul(poseMat, transMat_ycbObj2dopeObj[m]))
                        # print("Baselink frame pose")
                        # print(poseMat)

                        f_debug.write('PoseMat (baselink):\n')
                        writeMat2File(f_debug, poseMat)
                        f_debug.write('\n')

                        save_x = poseMat[0][3]
                        save_y = poseMat[1][3]
                        save_z = poseMat[2][3]

                        _, R_pose, _, _ = tf3.affines.decompose44(poseMat)

                        save_rw, save_rx, save_ry, save_rz = tf3.quaternions.mat2quat(R_pose)

                        f_debug.write('Baselink 6D: x, y, z, rw, rx, ry, rz:\n')
                        f_debug.write('{},{},{},{},{},{},{}\n'.format(save_x, save_y, save_z, save_rw, save_rx, save_ry, save_rz))

                        # Publish
                        pubs[m].publish(msg)
                        pub_dimension[m].publish(str(params['dimensions'][m]))

                        # Draw the cube
                        if None not in result['projected_points']:
                            points2d = []
                            for pair in result['projected_points']:
                                points2d.append(tuple(pair))
                            DrawCube(points2d, draw_colors[m])
                            f_debug.write('\nprojected_points\n')
                            writeMat2File(f_debug, result['projected_points'])
                            # print('Drawing...')
                            # print(result['projected_points'])
                        else:
                            # print("None in proj points. Not drawing.")
                            pass

                        # save pose txt
                        print('Pose found')
                    else:
                        f_debug.write('NAN\n')
                        print('Pose NOT found')

                print('Saving to ' + fn_PoseResult)
                with open(fn_PoseResult, 'a') as f:
                    f.write('{} {} {} {} {} {} {} {}\n'.format(
                            m, save_x, save_y, save_z, save_rw, save_rx, save_ry, save_rz
                        )
                    )
                
            # Publish the image with results overlaid
            pub_rgb_dope_points.publish(
                CvBridge().cv2_to_imgmsg(
                    np.array(im)[...,::-1], 
                    "bgr8"
                )
            )

            g_img = None
            g_scene = None

        else:
            # print("waiting...")
            pass
        rate.sleep()


if __name__ == "__main__":
    '''Main routine to run DOPE'''

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "config_pose.yaml"
    rospack = rospkg.RosPack()
    params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

    topic_cam = params['topic_camera']

    try :
        run_dope_node(params)
    except rospy.ROSInterruptException:
        pass
