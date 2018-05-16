#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

        #rospy.logwarn("State: " + str(self.state) )

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        min_distance = float('inf')
        nearest_index = 0

        px = pose.position.x
        py = pose.position.y

        for i, wp in enumerate(self.waypoints.waypoints):
            x = wp.pose.pose.position.x
            y = wp.pose.pose.position.y
            distance = math.sqrt((px - x)**2 + (py - y)**2)
            if distance < min_distance:
                nearest_index = i
                min_distance = distance
        return nearest_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x0, y0, x1, y1 = self.project_to_image_plane(light.pose.pose.position)
        #rospy.logwarn(self.project_to_image_plane(light.pose.pose.position) )
        
        if x0 == x1 or x0 < 0 or x1 > cv_image.shape[1] or \
            y0 == y1 or y0 < 0 or y1 > cv_image.shape[0]:
            return TrafficLight.UNKNOWN
        light_image = cv_image[y0:y1, x0:x1, :]        
        #cv2.imshow("123", light_image)
        #cv2.waitKey(1)
        #Get classification
        return self.light_classifier.get_classification(light_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None
        #light = None
        line_wp_idx = None
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        #rospy.logwarn("Car x: " + str(self.pose.pose.position.x) )  
        #rospy.logwarn("Car y: " + str(self.pose.pose.position.y) )  
        #if(self.pose):
        #    car_wp_idx = self.get_closest_waypoint(self.pose.pose)        
        #    cv_image1 = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #    cv_image = cv_image1.copy()

        #    hsv = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2HSV_FULL)
        #    h = hsv[:, :, 0]
        #    s = hsv[:, :, 1]     
        #    v = hsv[:, :, 2]             
        #    mask = np.zeros(h.shape, dtype=np.uint8)
        #    mask[((h < 10) | (h > 220)) & (s > 180) & (v > 180)] = 255
        #    kernel = np.ones((7,7),np.uint8)
        #    mask = cv2.erode(mask,kernel,iterations = 1)     
        #    mask = cv2.dilate(mask,kernel,iterations = 1)             
    
        #    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        #    contours.sort(key=cv2.contourArea, reverse=True)

        #   diff2 = None # now idx -> in future change distance position

        #    if len(contours) > 0 and cv2.contourArea(contours[0]) < 600:
                #cv2.imshow("1234", mask)
                #cv2.waitKey(1)                
        #        xx,yy,ww,hh = cv2.boundingRect(contours[0])
        #        x0 = int((xx + ww/2.0) - ww * 5)
        #        x1 = int((xx + ww/2.0) + ww * 5)
        #        y0 = int((yy + hh/2.0) - hh * 5)
        #        y1 = int((yy + hh/2.0) + hh * 5)

        #       diff2 = int((22.6 - math.sqrt(cv2.contourArea(contours[0]))) * 3 )
                #diff2 = (510 - cv2.contourArea(contours[0])) / 4.5

        #       if diff2 < 0:
        #           diff2 = 0

                #closest_light = self.lights[0]
                #closest_light.pose.pose.position.x = self.waypoints.waypoints[car_wp_idx + diff2].pose.pose.position.x
                #closest_light.pose.pose.position.y = self.waypoints.waypoints[car_wp_idx + diff2].pose.pose.position.y
                #####state = self.get_light_state(closest_light)
                #state = self.light_classifier.get_classification(light_image)                
                #rospy.logwarn(str(len(contours)) + " diff2=" + str(diff2)  + " state=" + str(state))
                #return car_wp_idx + diff2, state
                #rospy.logwarn("Area=" + str(cv2.contourArea(contours[0])) + " state=" + str(state))

                #light_image = cv_image[y0:y1, x0:x1, :]     
                #if x0 == x1 or x0 < 0 or x1 > cv_image.shape[1] or \
                #    y0 == y1 or y0 < 0 or y1 > cv_image.shape[0]:
                #    return -1, TrafficLight.UNKNOWN                
                
        #        return car_wp_idx + diff2, TrafficLight.RED                
        #    else:
        #       return -1, TrafficLight.UNKNOWN


        #return -1, TrafficLight.UNKNOWN
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose)
            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                stoplight_pose = Pose()
                stoplight_pose.position.x = line[0]
                stoplight_pose.position.y = line[1]
                temp_wp_idx = self.get_closest_waypoint(stoplight_pose)

                d = temp_wp_idx - car_wp_idx
             
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)

            #rospy.logwarn("Car index: " + str(car_wp_idx) )             
            #rospy.logwarn("Light index: " + str(line_wp_idx) )

            return line_wp_idx, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image
        """
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
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
