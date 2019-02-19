import timeit, time
import numpy as np

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

n_sample = 20000
n_feature = 500
X = np.random.rand(n_sample, n_feature)
# use same number of hidden layers for different frameworks
H1, H2 = 400, 400
import torch
from torch.autograd import Variable
print ('torch version: ', torch.__version__)

model = torch.nn.Sequential(
    torch.nn.Linear(n_feature, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, 1)
)

X_torch = Variable(torch.from_numpy(X).type(torch.FloatTensor))

t = time.time()
model(X_torch).data.numpy()
print ('time for torch: %s ms' % ((time.time()-t) * 1000))

# export pytorch model to ONNX
dummy_input = Variable(torch.randn(n_sample, n_feature))
torch.onnx.export(model, dummy_input, "MLP.onnx")

import keras
from keras import models, layers
print ('keras version: ', keras.__version__)

network = models.Sequential()
network.add(layers.Dense(H1, activation='relu', input_shape=(n_feature,)))
network.add(layers.Dense(H2, activation='relu'))
network.add(layers.Dense(1, activation=None))

t = time.time()
network.predict(X)
print ('time for keras: %s ms' % ((time.time() - t) * 1000))

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
print ('tensorflow version: ', tf.__version__)

tf.reset_default_graph()
X_tf = tf.placeholder(tf.float32, shape=(None, n_feature), name='X')
hidden1 = fully_connected(X_tf, H1, scope="hidden1")
hidden2 = fully_connected(hidden1, H2, scope='hidden2')
out = fully_connected(hidden2, 1, scope='outputs', activation_fn=None)

init = tf.global_variables_initializer()

t = time.time()
with tf.Session() as sess:
    init.run()
    out.eval(feed_dict={X_tf: X})
    print('time for tensorflow: %s ms' % ((time.time() - t)*1000))

