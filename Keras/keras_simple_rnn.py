from keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, RNN
import keras
import numpy as np
import time

model = Sequential()
model.add(SimpleRNN(32, input_shape=(1, 150528), activation='tanh'))

X = np.random.rand(1, 1, 150528)
t = time.time()
print (model.predict(X))
print ('time for Keras Simple RNN: %s ms' % ((time.time()-t) * 1000))

model.save('keras_rnn.h5')




