import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import time
from util import save_tf_model

n_feature = 150528
hidden_size = 32


def tensorflow_rnn():

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

  input_data = tf.placeholder(tf.float32, shape=(1, 1, n_feature), name='input')
  input_data = tf.reshape(input_data, [1, 1, 150528])
  # defining initial state
  initial_state = rnn_cell.zero_state(1, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  return outputs, state


def eval_tensorflow_rnn():
  X = np.random.rand(1, 1, n_feature)
  with tf.Session() as sess:
    outputs, state = tensorflow_rnn()
    sess.run(tf.global_variables_initializer())
    save_tf_model.save_to_ckpt(sess, 'D:\PerfCompare\\tensorflow\RNN\\tf_RNN.ckpt')
    save_tf_model.save_to_frozen(sess, 'D:\PerfCompare\\tensorflow\RNN\\tf_RNN.pb', 'rnn/transpose_1')
    t = time.time()
    pred = sess.run([outputs, state], feed_dict={'input:0': X})
    print('time for tensorflow Dynamic RNN: %s ms' % ((time.time() - t) * 1000))
    print(pred)

if __name__ == '__main__':
  eval_tensorflow_rnn()

  


