# This file evaluate the performance of models(mobilenetV1, SimpleRNN) on PyTorch, Tensorflow and Keras on CPU

import tensorflow as tf
import torch
from torch.autograd import Variable
import numpy as np

from tensorflow_models.mobilenet.mobilenet import tf_mobilenet
from tensorflow_models.RNN.tf_RNN import tensorflow_rnn
from pytorch_models.torch_mobilenet import torch_mobilenet
from pytorch_models.pytorch_rnn import torch_rnn
from Keras_models.keras_mobilenet import keras_mobilenet
from Keras_models.keras_simple_rnn import keras_rnn
from util.perf_evaluate import *

def evaluate_tensorflow_mobilenet():
  with tf.Graph().as_default() as graph:
    with tf.device('/cpu:0'):
      image_size = 224
      inputs = tf.Variable(tf.random_normal([1,
                                             image_size,
                                             image_size, 3],
                                            dtype=tf.float32,
                                            stddev=1e-1))
      logits, _ = tf_mobilenet(inputs, is_training=False)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        time_run_mobilenet('tensorflow', session=sess, target=logits, info_string='tensorflow mobilenet')
          
def evaluate_torch_mobilenet():
  model = torch_mobilenet()
  input = Variable(torch.rand(1, 3, 224, 224))
  time_run_mobilenet('pytorch', model=model, input=input, info_string='pytorch mobilenet')
  
def evaluate_keras_mobilenet():
  model = keras_mobilenet()
  input = np.random.rand(1, 224, 224, 3)
  time_run_mobilenet('keras', model=model, input=input, info_string='keras mobilenet')

def evaluate_torch_RNN():
  model = torch_rnn()
  input = torch.randn(1, 1, 150528)
  time_run_mobilenet('pytorch', model=model, input=input, info_string='pytorch RNN')
  
def evaluate_tensorflow_RNN():
  X = np.random.rand(1, 1, 150528)
  with tf.Session() as sess:
    outputs, _ = tensorflow_rnn()
    sess.run(tf.global_variables_initializer())
    time_run_mobilenet('tensorflow', session=sess, target=outputs, feed_dict={'input:0': X}, info_string='tensorflow RNN')
    
def evaluate_keras_RNN():
  model = keras_rnn()
  input = np.random.rand(1, 1, 150528)
  time_run_mobilenet('keras', model=model, input=input, info_string='keras RNN')
  
if __name__ == '__main__':
  evaluate_tensorflow_mobilenet()
  evaluate_torch_mobilenet()
  evaluate_keras_mobilenet()
  evaluate_torch_RNN()
  evaluate_tensorflow_RNN()
  evaluate_keras_RNN()
  
