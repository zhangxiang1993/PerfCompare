import keras
model = keras.models.Sequential()
model.add(keras.layers.SimpleRNN(5, input_shape=(1, 2), batch_input_shape=[1, 1, 2], stateful=True))