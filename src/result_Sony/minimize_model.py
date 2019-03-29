import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib

saver = tf.train.import_meta_graph('./model.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./model.ckpt")

output_node_names="network_output"
output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
)

output_graph="./squeeze-model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()

inputGraph = tf.GraphDef()
with tf.gfile.Open('./squeeze-model.pb', "rb") as f:
  data2read = f.read()
  inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
              inputGraph,
              ["inputTensor"], # an array of the input node(s)
              ["output/softmax"], # an array of output nodes
              tf.int32.as_datatype_enum)

# Save the optimized graph'test.pb'
f = tf.gfile.FastGFile('OptimizedGraph.pb', "w")
f.write(outputGraph.SerializeToString())