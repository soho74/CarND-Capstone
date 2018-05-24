import cv2
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
import rospy

class TLClassifier(object):
    def __init__1(self):
        print(tf.__version__)

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default() as graph:
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile("light_classification/tlc_model/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

          with tf.Session() as sess:
            self.tensor_dict = {}
            self.tensor_dict['detection_scores'] = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            self.tensor_dict['detection_classes'] = tf.get_default_graph().get_tensor_by_name('detection_classes:0')    
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.sess = tf.Session(graph=graph)                          

            
    def get_classification1(self, image):

        image = cv2.resize(image[:,:,::-1],(200, 200), cv2.INTER_CUBIC)
        final_image = np.expand_dims(image, 0)

        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: final_image})

        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        score = output_dict['detection_scores'][0]

        if score > 0.4:
            num = output_dict['detection_classes'][0] -1
        else:
            num = 6

        print("                                                                                  ", num, score)

        if num == 6:
            return 4
        else:
            return num % 3
        return num

    def __init__(self):
        #TODO load classifier        
        with tf.gfile.GFile('light_classification/tlc_model/tlc_model.pb', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='tlc',
                op_dict=None,
                producer_op_list=None)
            self.image = graph.get_tensor_by_name('tlc/images:0')
            self.keep_prob = graph.get_tensor_by_name('tlc/keep_prob:0')
            self.pred = graph.get_tensor_by_name('tlc/predict:0')
            self.sess = tf.Session(graph=graph)
            self.counter = 0
            
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        image = cv2.resize(image, dsize=(40, 40))
        image = cv2.normalize(image.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        feed_dict = {self.image: [image],
                     self.keep_prob: 1.}
        lid = self.sess.run(self.pred, feed_dict=feed_dict)[0]
        lid = 4 if lid == 3 else lid
        
        # For debugging
        light = {2: 'green', 0: 'red', 1: 'yellow', 4: 'unknown'}
        self.counter += 1
        print self.counter, light[lid]
        
        return lid
