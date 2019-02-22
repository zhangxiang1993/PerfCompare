import tensorflow as tf
from PIL import Image
import numpy as np
import time
import argparse

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)

    return graph

def HWC_to_CHW(data):
    if(len(data.shape) == 3):
        return tf.transpose(data, [2, 0, 1])
    if(len(data.shape) == 4):
        return tf.transpose(data, [0, 3, 1, 2])

def print_op_names(graph):
    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1, help="how many times to evaluate")

models = [
    {"modelPath": "E:\Pycharm Project\PerfCompare\mobilenet_v2_1.0_224\mobilenet_v2_1.0_224_frozen.pb",
     "input": 'prefix/input:0',
     "output": 'prefix/MobilenetV2/Predictions/Reshape_1:0'},
    {"modelPath": "E:\Pycharm Project\PerfCompare\MobileNet.pb",
     "input": 'prefix/data:0',
     "output": 'prefix/mobilenetv20_output_flatten0_reshape0:0'},
    {"modelPath": "E:\Pycharm Project\PerfCompare\MobileNet_TF1.8.pb",
     "input": 'prefix/data:0',
     "output": 'prefix/mobilenetv20_output_flatten0_reshape0:0'},
    {'modelPath': 'E:\Pycharm Project\PerfCompare\mobilenet_v1_1.0_224\mobilenet_v1_1.0_224_frozen.pb',
     "input": 'prefix/input:0',
     "output": 'prefix/MobilenetV1/Predictions/Reshape_1:0'
     },
    {'modelPath': 'D:\PerfCompare\\tf_mobilenet\mobilenet_tf.pb',
     "input": 'import/MobileNet/Reshape:0',
     "output": 'import/MobileNet/Predictions/Softmax:0'
     }
]


if __name__ == '__main__':
    model_idx = 4
    total_start = time.time()
    # Loading Model
    start_time = time.time()
    graph = load_graph(models[model_idx]["modelPath"])
    print_op_names(graph)
    end_time = time.time()
    print('\nLoad: {:.4f} millionseconds'.format((end_time - start_time) * 1000))

    x = graph.get_tensor_by_name(models[model_idx]["input"])
    y = graph.get_tensor_by_name(models[model_idx]["output"])

    imagePath = 'E:\Pycharm Project\PerfCompare\kitten_224.png'
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
    # image_data = tf.gfile.FastGFile(image, 'rb').read()
    # img = Image.open(imagePath)
    # image_data = np.asarray(img, dtype="float32")
    # input_data = np.empty((1, 224, 224, 3), dtype="float32")
    #
    #
    # input_data[0,:,:,:] = image_data[:,:,0:3]
    # input_data = np.transpose(input_data, (0, 3, 1, 2))
    
    input_data = np.random.rand(1, 224, 224, 3)

    start_time = time.time()
    with tf.Session(graph=graph) as sess:
        predictions = sess.run(y, {x: input_data})
    end_time = time.time()
    print('\nPredict: {:.4f} millionseconds'.format((end_time - start_time) * 1000))
    #print (predictions[0])
    total_end = time.time()
    print('\ntotal: {:.4f} millionseconds'.format((total_end - total_start) * 1000))


