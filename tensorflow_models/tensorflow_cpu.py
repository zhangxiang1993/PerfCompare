import tensorflow as tf
from PIL import Image
import numpy as np
import time
import argparse

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    tf.reset_default_graph()
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)

    return graph

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
    {'modelPath': 'D:\PerfCompare\\tensorflow\mobilenet_tf.pb',
     "input": 'import/MobileNet/Reshape:0',
     "output": 'import/MobileNet/Predictions/Softmax:0'
     },
    {'modelPath': 'D:\PerfCompare\\tensorflow\\tf_RNN.pb',
     "input": 'import/X:0',
     "output": 'import/rnn/transpose_1:0'
     }
]

def evaluate_frozen_graph():
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

    input_data = np.random.rand(1, 224, 224, 3)
    # input_data = np.random.rand(1, 1, 150528)

    with tf.Session(graph=graph) as sess:
        start_time = time.time()
        predictions = sess.run(y, {x: input_data})
        end_time = time.time()
        print('\nPredict: {:.4f} millionseconds'.format((end_time - start_time) * 1000))
    print (predictions[0])
    total_end = time.time()
    print('\ntotal: {:.4f} millionseconds'.format((total_end - total_start) * 1000))


def eval_checkpoint():
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('mobilenet_tf_forward.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        print_op_names(graph)
        input = graph.get_tensor_by_name('MobileNet/Reshape:0')
        output = graph.get_tensor_by_name('MobileNet/Predictions/Softmax:0')

        input_data = np.random.rand(1, 224, 224, 3)

        start_time = time.time()
        result = sess.run(output, feed_dict={input: input_data})
        end_time = time.time()
        print('\nPredict: {:.4f} millionseconds'.format((end_time - start_time) * 1000))
        print(result)

if __name__ == '__main__':
    # eval_checkpoint()
    evaluate_frozen_graph()

