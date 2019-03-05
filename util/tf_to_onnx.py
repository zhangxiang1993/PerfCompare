import winmltools
import tensorflow
import tf2onnx
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph
import onnx
import tensorflow as tf
from save_tf_model import print_op_names

def tf_to_onnx(filename, outputs, name=""):
  graph_def = graph_pb2.GraphDef()
  with open(filename, 'rb') as file:
      graph_def.ParseFromString(file.read())
  g = tf.import_graph_def(graph_def, name='')
  with tf.Session(graph=g) as sess:
    print_op_names(sess.graph)
    g = tf2onnx.tfonnx.process_tf_graph(sess.graph, output_names=outputs)
    doc = 'converted_from {}'.format(name) if name is not None else ''
    converted_model = g.make_model(doc)
  return converted_model

def save_onnx(onnx_model, destination):
  winmltools.utils.save_model(onnx_model, destination)
    
if __name__ == "__main__":
  filename = "D:\\PerfCompare\\models\\mobilenet_v1_1.0_224_frozen.pb"
  name = "tensorflow"
  converted_model = tf_to_onnx(filename, ["MobilenetV1/Predictions/Reshape_1:0"], name)
  save_onnx(converted_model, "D:\\PerfCompare\\models\\mobilenet_converted_from_tf.onnx")
  