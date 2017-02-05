import os

import tensorflow as tf
import pandas as pd  # for CSV manipulation

import utils


def load_graph(sess, model_filename,
               jpeg_data_tensor_name='DecodeJpeg/contents:0',
               final_tensor_name='retrained_result:0'):
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        jpeg_data_tensor, final_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[jpeg_data_tensor_name, final_tensor_name])
        )
        return sess.graph, jpeg_data_tensor, final_tensor


def read_labels_list():
    return list(map(lambda label: label.strip(), tf.gfile.GFile('./output/labels.txt').readlines()))


def main():
    test_dir = './data/test_stg1'
    utils.ensure_dir_exists(test_dir)
    labels = read_labels_list()
    result_df = pd.DataFrame(columns=labels)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        graph, jpeg_data_tensor, final_tensor = load_graph(sess, './output/output_graph.pb')
        file_glob = os.path.join(test_dir, '*.jpg')
        image_paths = tf.gfile.Glob(file_glob)
        for image_path in image_paths:
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            prediction = sess.run([final_tensor], {jpeg_data_tensor: image_data})
            image_name = os.path.basename(image_path).strip()
            result_df.loc[image_name] = prediction[0].flatten()
    result_df.to_csv('./output/stg1.csv', index_label='image', float_format='%.16f')


if __name__ == '__main__':
    main()
