import tensorflow as tf
from tensorflow.python.framework import graph_util
from mobilenet import mobilenet, mobilenet_arg_scope
import numpy as np
slim = tf.contrib.slim

def save_mobilenet():
  with tf.Graph().as_default() as graph:
    with tf.device('/cpu:0'):
      image_size = 224
      inputs = tf.Variable(tf.random_normal([1,
                                             image_size,
                                             image_size, 3],
                                            dtype=tf.float32,
                                            stddev=1e-1))
      with slim.arg_scope(mobilenet_arg_scope()):
        logits, _ = mobilenet(inputs, is_training=False)
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['MobileNet/Predictions/Softmax'])
          print(sess.run(logits))
          with tf.gfile.GFile('mobilenet_tf.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
          
if __name__ == '__main__':
  save_mobilenet()
        
      
      
      
