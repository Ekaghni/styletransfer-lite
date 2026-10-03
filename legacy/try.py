import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 500
NUM_STYLES = 26

# Load the frozen graph
def load_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    return graph

def stylize_image(image_path, model_path, style_index, strength=100):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Prepare style values
    other_styles = ((100.0 - strength) / (NUM_STYLES - 1)) / 100.0
    style_vals = np.full(NUM_STYLES, other_styles, dtype=np.float32)
    style_vals[style_index] = strength / 100.0

    # Load graph and run inference
    graph = load_graph(model_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('input:0')
        style_tensor = graph.get_tensor_by_name('style_num:0')
        output_tensor = graph.get_tensor_by_name('transformer/expand/conv3/conv/Sigmoid:0')

        output = sess.run(output_tensor, feed_dict={
            input_tensor: img_array,
            style_tensor: style_vals
        })

    # Convert output to image
    output = np.clip(output[0] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

if __name__ == '__main__':
    model_path = 'stylize_quantized.pb'
    image_path = 'girl.png'
    style_index = 0  # 0-25 for different styles
    strength = 100   # 0-100

    result = stylize_image(image_path, model_path, style_index, strength)
    result.save('output.png')
    print('Saved: output.png')