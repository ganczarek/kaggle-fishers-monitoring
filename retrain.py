#
# This is simplified version of
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
#
import tensorflow as tf

import utils
from inception_v3 import InceptionV3


def cache_bottlenecks(training_data, inception_v3):
    with tf.Session() as sess:
        for label_name, label_data in training_data.items():
            label_dir = label_data['dir']
            for category in ['training', 'testing', 'validation']:
                category_list = label_data[category]
                for file_name in category_list:
                    inception_v3.get_or_create_bottleneck(sess, label_dir, file_name)


def main():
    summaries_dir = utils.create_summaries_dir()
    inception_v3 = InceptionV3()
    training_data = utils.get_training_data()
    cache_bottlenecks(training_data, inception_v3)


if __name__ == '__main__':
    main()
