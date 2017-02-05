import hashlib
import os

import tensorflow as tf
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def ensure_dir_exists(directory):
    if not tf.gfile.Exists(directory):
        print("Creating", directory)
        tf.gfile.MakeDirs(directory)


def ensure_dir_empty(directory):
    if tf.gfile.Exists(directory):
        print("Deleting", directory)
        tf.gfile.DeleteRecursively(directory)


def create_summaries_dir(summaries_directory='/tmp/kaggle_fishers_monitoring_retrain_logs'):
    ensure_dir_exists(summaries_directory)
    ensure_dir_empty(summaries_directory)
    return summaries_directory


def get_training_data(test_data_dir='./data/test_stg1',
                      train_data_dir='./data/train',
                      validation_percentage=10,
                      testing_percentage=10):
    if not tf.gfile.Exists(test_data_dir):
        raise Exception(test_data_dir + " does not exist!")
    if not tf.gfile.Exists(train_data_dir):
        raise Exception(train_data_dir + " does not exist!")

    label_dirs = [x[0] for x in tf.gfile.Walk(train_data_dir)][1:]  # first entry is top directory, ignore
    training_data = {}
    for label_dir in label_dirs:
        label_name = label_dir.split("/")[-1]
        file_glob = os.path.join(label_dir, '*.jpg')
        file_paths = tf.gfile.Glob(file_glob)

        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_paths:
            base_name = os.path.basename(file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(file_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        training_data[label_name] = {
            'dir': label_dir,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    print_training_data(training_data)
    return training_data


def print_training_data(training_data):
    for label_name, label_data in training_data.items():
        training_images = len(label_data['training'])
        testing_images = len(label_data['testing'])
        validation_images = len(label_data['validation'])
        print(label_name + ":\t", "training set =", training_images, "testing set =", testing_images,
              "validation set =", validation_images)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
