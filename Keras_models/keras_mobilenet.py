from keras.applications import mobilenet
import numpy as np
import time
import keras

def keras_mobilenet():
  model = mobilenet.MobileNet(input_shape=(224, 224, 3))
  return model

def eval_keras_mobilenet():
  model = keras_mobilenet()
  X = np.random.rand(1, 224, 224, 3)

  t = time.time()
  model.predict(X)
  print('time for Keras MobineNet: %s ms' % ((time.time() - t) * 1000))

  model.save('keras_mobilenet.h5')

if __name__ == '__main__':
  eval_keras_mobilenet()
