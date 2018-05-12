import cv2
import tensorflow as tf
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
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

