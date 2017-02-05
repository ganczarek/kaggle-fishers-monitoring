import os
import tarfile
import urllib.request

import numpy as np
import tensorflow as tf

import utils


class InceptionV3:
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    graph_def_file_name = 'classify_image_graph_def.pb'
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    jpeg_data_tensor_name = 'DecodeJpeg/contents:0'
    resized_input_tensor_name = 'ResizeBilinear:0'
    bottleneck_tensor_size = 2048

    MODEL_INPUT_WIDTH = 299
    MODEL_INPUT_HEIGHT = 299
    MODEL_INPUT_DEPTH = 3
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

    def __init__(self, model_dir='./data/inception-v3'):
        self.model_dir = model_dir
        self.__maybe_get_inception_v3_model()
        self.graph, self.bottleneck_tensor, self.jpeg_data_tensor, self.resized_input_tensor = self.__create_graph()

    def __maybe_get_inception_v3_model(self):
        utils.ensure_dir_exists(self.model_dir)
        filename = self.data_url.split('/')[-1]
        file_path = os.path.join(self.model_dir, filename)
        if not tf.gfile.Exists(file_path):
            print('Downloading', filename)
            file_path, _ = urllib.request.urlretrieve(self.data_url, file_path)
            stat_info = os.stat(file_path)
            print('Successfully downloaded', filename, stat_info.st_size, 'bytes.')
        tarfile.open(file_path, 'r:gz').extractall(self.model_dir)

    def __create_graph(self):
        with tf.Session() as sess:
            model_filename = os.path.join(self.model_dir, self.graph_def_file_name)
            with tf.gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                        self.bottleneck_tensor_name, self.jpeg_data_tensor_name, self.resized_input_tensor_name])
                )
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

    def get_or_create_bottleneck(self, sess, label_dir, file_name, bottleneck_dir='/tmp/bottlenecks'):
        utils.ensure_dir_exists(bottleneck_dir)
        image_path = os.path.join(label_dir, file_name)
        bottleneck_path = os.path.join(bottleneck_dir, file_name + ".txt")
        if tf.gfile.Exists(bottleneck_path):
            bottleneck_values = self.__read_bottleneck_values_from_file(bottleneck_path)
        else:
            bottleneck_values = self.__run_bottleneck_on_image(sess, image_path, bottleneck_path)
        return bottleneck_values

    def __run_bottleneck_on_image(self, sess, image_path, bottleneck_path):
        print('Creating bottleneck at', bottleneck_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = sess.run(self.bottleneck_tensor, {self.jpeg_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w+') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
        return bottleneck_values

    @staticmethod
    def __read_bottleneck_values_from_file(bottleneck_path):
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        return [float(x) for x in bottleneck_string.split(',')]
