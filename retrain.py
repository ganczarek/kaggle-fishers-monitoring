#
# This is simplified version of
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
#
import random

import numpy as np
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


def get_random_cached_bottlenecks(sess, training_data, batch_size, category, inception_v3):
    labels = list(training_data.keys())
    bottlenecks = []
    ground_truths = []
    file_names = []
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(batch_size):
        label_index = random.randrange(len(labels))
        label_name = labels[label_index]
        image_name = random.choice(training_data[label_name][category])
        bottleneck = inception_v3.get_or_create_bottleneck(sess, training_data[label_name]['dir'], image_name)

        ground_truth = np.zeros(len(labels), dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        file_names.append(image_name)
    return bottlenecks, ground_truths, file_names


def add_training_ops(class_count, inception_v3, learning_rate, final_tensor_name='retrained_result'):
    with tf.name_scope('retrain_input'):
        # first dimension is a batch size
        bottleneck_input = tf.placeholder_with_default(inception_v3.bottleneck_tensor,
                                                       shape=[None, inception_v3.bottleneck_tensor_size],
                                                       name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        with tf.name_scope('weights'):
            initial_values = tf.truncated_normal([inception_v3.bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_values, name='final_weights')
            utils.variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            utils.variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def main():
    # training parameters
    training_steps = 4000
    batch_size = 1000
    learning_rate = 0.01

    # import Inception-v3 model
    inception_v3 = InceptionV3()
    training_data = utils.get_training_data()
    cache_bottlenecks(training_data, inception_v3)

    train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor = (
        add_training_ops(len(training_data), inception_v3, learning_rate)
    )

    # Merge all the summaries and write them out to summaries dir
    summaries_dir = utils.create_summaries_dir()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', inception_v3.graph)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # start training
        for i in range(training_steps):
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                sess, training_data, batch_size, 'training', inception_v3)
            # Feed the bottlenecks and ground truth into the graph, and run a training step.
            # Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)


if __name__ == '__main__':
    main()
