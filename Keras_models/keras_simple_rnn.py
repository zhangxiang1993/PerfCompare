from keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, RNN
import keras
import numpy as np
import time

def keras_rnn():
  model = Sequential()
  model.add(SimpleRNN(32, input_shape=(1, 150528), activation='tanh'))
  return model

def eval_keras_rnn():
  model = keras_rnn()
  X = np.random.rand(1, 1, 150528)
  t = time.time()
  print (model.predict(X))
  print ('time for Keras Simple RNN: %s ms' % ((time.time()-t) * 1000))

  # save keras model
  model.save('keras_rnn.h5')

if __name__ == '__main__':
  eval_keras_rnn()




