#-*- coding:utf-8 -*-  
import os, argparse  
import tensorflow as tf  
from tensorflow.python.framework import graph_util  


def freeze_graph(model_folder):  
    checkpoint = tf.train.get_checkpoint_state(model_folder)  
    input_checkpoint = checkpoint.model_checkpoint_path  
      
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])  
    output_graph = absolute_model_folder + "/tlc_model.pb"  
    output_node_names = "predict"  
    saver = tf.train.import_meta_graph(
        input_checkpoint + '.meta', 
        clear_devices=True)  
  
    # We retrieve the protobuf graph definition  
    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
  
    #We start a session and restore the graph weights  
    with tf.Session() as sess:  
        saver.restore(sess, input_checkpoint)  
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names.split(",") 
        )   
  
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(output_graph, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
        print("%d ops in the final graph." % len(output_graph_def.node))  
  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_folder", type=str, help="Model folder to export")  
    args = parser.parse_args()  
    freeze_graph(args.model_folder)  


