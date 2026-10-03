import tensorflow as tf

with tf.io.gfile.GFile('stylize_quantized.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name='')
    wrapped = tf.compat.v1.wrap_function(_imports_graph_def, [])
    return wrapped.prune(
        tf.nest.map_structure(wrapped.graph.as_graph_element, inputs),
        tf.nest.map_structure(wrapped.graph.as_graph_element, outputs)
    )

concrete_func = wrap_frozen_graph(
    graph_def,
    inputs=['input:0', 'style_num:0'],
    outputs=['transformer/expand/conv3/conv/Sigmoid:0']
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Post-training quantization
tflite_model = converter.convert()

with open('stylize.tflite', 'wb') as f:
    f.write(tflite_model)