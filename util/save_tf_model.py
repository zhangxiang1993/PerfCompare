import tensorflow as tf
from tensorflow.python.framework import graph_util

def save_to_ckpt(session, destination):
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(session, destination)

def save_to_frozen(session, destination, output_node_name):
    constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [output_node_name])
    with tf.gfile.GFile(destination, mode='wb') as f:
        f.write(constant_graph.SerializeToString())