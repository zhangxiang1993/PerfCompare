from keras.applications import mobilenet
import numpy as np
import time

model = mobilenet.MobileNet(input_shape=(224, 224, 3))
X = np.random.rand(1, 224, 224, 3)

t = time.time()
model.predict(X)
print('time for Keras MobineNet: %s ms' % ((time.time() - t) * 1000))
