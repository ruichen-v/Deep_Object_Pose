#!/usr/bin/env python
#
# @Author Rui Chen
# @Brief Read YCB adv data and feed to dope
# @ Feb. 2019
#

from __future__ import print_function
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
import numpy as np
import cv2
import os
from dope.msg import AdvYCBFrame as Adv_msg

from PIL import Image

data_folder = '/home/ruic/Documents/RESEARCH/4PROGRESS/robust_obj_detect/data/ycb_adv_dataset'
topic = '/dope/webcam_rgb_raw'  # topic for publishing

class Input_t:    

    def __init__(self):
        self.scene_id_list = range(1, 2)
        self.scene_setting_list = ['B','D']
        # , 'D', 'D1', 'D2', 'O1', 'O2', 'O3']
        self.scene_names = []
        self.current_scene_id = 0
        self.done = False
        for scene_id in self.scene_id_list:
            for setting in self.scene_setting_list:
                self.scene_names.append('exp{:0>3d}_{}'.format(scene_id, setting))
        print('Totally {} scenes collected.'.format(len(self.scene_names)))

    def next(self):
        """
            returns a cv2 img
        """
        scene = self.scene_names[self.current_scene_id]
        fn = os.path.join(data_folder, scene, 'scenergb.jpg')
        frame = cv2.imread(fn)
        if frame is None:
            print('Cannot read file {}'.format(fn))
        else:
            print('Openning file {}'.format(fn))
        self.current_scene_id += 1
        if self.current_scene_id == len(self.scene_names):
            self.done = True
        self.current_scene_id = self.current_scene_id % len(self.scene_names)
        return scene, frame, self.done

Input = Input_t()

def publish_images(freq=0.5):
    rospy.init_node('dope_webcam_rgb_raw', anonymous=True)
    images_publisher = rospy.Publisher(topic, Adv_msg, queue_size=10)
    rate = rospy.Rate(freq)

    print ("Publishing images from source {} to topic '{}'...".format(
            data_folder, 
            topic
            )
    )
    print ("Ctrl-C to stop")
    
    done = False
    waiting = False
    first_skipped = False
    while not rospy.is_shutdown() and not done:
        
        if images_publisher.get_num_connections() > 0:
            waiting = False
            if not first_skipped:
                first_skipped = True
                continue
            scene, frame, done = Input.next()
        else:
            if not waiting:
                waiting = True
                print('Waiting for connection...')
            continue

        if frame is not None:
            out_adv_frame = Adv_msg()
            out_adv_frame.image = CvBridge().cv2_to_imgmsg(frame, "bgr8")
            out_adv_frame.scene_name = scene
            images_publisher.publish(out_adv_frame)
        else:
            print('Frame None')

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_images()
    except rospy.ROSInterruptException:
        pass