import tensorflow as tf

saver = tf.train.import_meta_graph('./model.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./model.ckpt")

output_node_names="DepthToSpace"
output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
)

output_graph="./squeeze-model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()