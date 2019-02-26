import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import time

n_feature = 150528
hidden_size = 32
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

input_data = tf.placeholder(tf.float32, shape=(1, 1, n_feature), name='X')
input_data = tf.reshape(input_data, [1, 1, 150528])
# defining initial state
initial_state = rnn_cell.zero_state(1, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)

X = np.random.rand(1, 1, n_feature)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # for op in sess.graph.get_operations():
    # print(op.name)
  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['rnn/transpose_1'])
  with tf.gfile.GFile('tf_RNN.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())
  t = time.time()
  pred = sess.run([outputs, state], feed_dict={input_data: X})
  print('time for tensorflow Dynamic RNN: %s ms' % ((time.time() - t) * 1000))
  print(pred)

  


