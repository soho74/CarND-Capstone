#!/usr/bin/env python 

import numpy as np
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight

from sensor_msgs.msg import Image as msgImage
from cv_bridge import CvBridge
import Tkinter
import Image, ImageTk
import tkMessageBox
import cv2
from Tkinter import NW
import yaml

class Dashboard(object):
    def __init__(self):
        self.pose = None
        self.stopline_wp_idx = -1        
        self.waypoints_2d = None
        self.waypoint_tree = None
        

        self.transformed_pose = None
        self.has_image = False
        self.camera_image = None
        self.bridge = CvBridge()       

        self.image_to_display = None
        
        self.min_x = 0
        self.min_y = 0
        self.max_x = 1
        self.max_y = 1
        self.message_index = 0

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        self.stop_line_positions = self.config['stop_line_positions']
        self.slp_transformed = None

        #self.base_waypoints = None
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub6 = rospy.Subscriber('/image_color', msgImage, self.image_cb)
        #rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        top = Tkinter.Tk()

        self.C = Tkinter.Canvas(top, bg="white", height=400, width=800)

        self.C.pack()
        top.mainloop()
        
        

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        self.message_index += 1
        if (self.message_index % 20):

            if (self.slp_transformed):
                for light in self.slp_transformed:
                    self.C.create_oval(light[0]-5, light[1]-5,
                                    light[0]+5, light[1]+5, fill='yellow')
                

            self.transformed_pose = (450 + 300  * (self.pose.pose.position.x - self.min_x) / (self.max_x - self.min_x), 
                            10 + 300 * (self.pose.pose.position.y - self.min_y) / (self.max_y - self.min_y))
            
            self.C.create_oval(self.transformed_pose[0]-5, self.transformed_pose[1]-5,
                            self.transformed_pose[0]+5, self.transformed_pose[1]+5, fill='red')
 


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

        self.min_x = min([w.pose.pose.position.x for w in self.base_waypoints.waypoints])
        self.min_y = min([w.pose.pose.position.y for w in self.base_waypoints.waypoints])
        self.max_x = max([w.pose.pose.position.x for w in self.base_waypoints.waypoints])
        self.max_y = max([w.pose.pose.position.y for w in self.base_waypoints.waypoints])

        self.transformed = [(450 + 300 * (w.pose.pose.position.x - self.min_x) / (self.max_x - self.min_x), 
                                10 + 300 * (w.pose.pose.position.y - self.min_y) / (self.max_y - self.min_y)) for w in self.base_waypoints.waypoints]
        
        self.C.create_polygon([e for l in self.transformed for e in l], fill='', outline='black', width = 10)

        self.slp_transformed = [(450 + 300 * (w[0] - self.min_x) / (self.max_x - self.min_x), 
                                10 + 300 * (w[1] - self.min_y) / (self.max_y - self.min_y)) for w in self.stop_line_positions]
     

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        self.has_image = True
        self.camera_image = msg
        
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #x0, y0, x1, y1 = self.project_to_image_plane(light.pose.pose.position)
        
        #if x0 == x1 or x0 < 0 or x1 > cv_image.shape[1] or \
        #    y0 == y1 or y0 < 0 or y1 > cv_image.shape[0]:
        #    return TrafficLight.UNKNOWN
        #light_image = cv_image[y0:y1, x0:x1, :]    


        small = cv2.resize(cv_image, (0,0), fx=0.5, fy=0.5) 

        b,g,r = cv2.split(small)
        small = cv2.merge((r,g,b))

        im = Image.fromarray(small)
        self.image_to_display = ImageTk.PhotoImage(image=im)

        self.C.create_image((0, 0), image = self.image_to_display, anchor=NW)
       
    def project_to_image_plane(self, point_in_world):
        # reference: https://discussions.udacity.com/t/focal-length-wrong/358568/23
        fx = 2574
        fy = 2744
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
                                        
        x_offset = (image_width / 2) - 30
        y_offset = image_height + 50 
        corner_offset = 1.5

        try:
            now = rospy.Time.now()
            self.listener.waitForTransform(
                "/base_link", "/world", self.pose.header.stamp, rospy.Duration(1.0))
            transT, rotT = self.listener.lookupTransform(
                "/base_link", "/world", self.pose.header.stamp)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return 0, 0, 0, 0

        RT = np.mat(self.listener.fromTranslationRotation(transT, rotT))
        point_3d = np.mat([[point_in_world.x], 
                            [point_in_world.y],
                            [point_in_world.z], 
                            [1.0]])
        point_3d_vehicle = (RT * point_3d)[:-1, :]
        camera_height_offset = 1.1
        camera_x = -point_3d_vehicle[1]
        camera_y = -(point_3d_vehicle[2] - camera_height_offset)
        camera_z = point_3d_vehicle[0]

        x0 = int((camera_x - corner_offset) * fx / camera_z) + x_offset
        y0 = int((camera_y - corner_offset) * fy / camera_z) + y_offset
        x1 = int((camera_x + corner_offset) * fx / camera_z) + x_offset
        y1 = int((camera_y + corner_offset) * fy / camera_z) + y_offset
        return x0, y0, x1, y1
          
if __name__ == '__main__':
    try:
        rospy.init_node("dashboard")
        Dashboard()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start dashboard node.')

