#
# This is simplified version of
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
#
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

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


def get_all_bottlenecks(sess, training_data, category, inception_v3):
    labels = list(training_data.keys())
    bottlenecks = []
    ground_truths = []
    file_names = []
    for label_index, label_name in enumerate(training_data.keys()):
        label_data = training_data[label_name]
        for image_name in label_data[category]:
            bottleneck = inception_v3.get_or_create_bottleneck(sess, label_data['dir'], image_name)

            ground_truth = np.zeros(len(labels), dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            file_names.append(os.path.join(label_data['dir'], image_name))
    return bottlenecks, ground_truths, file_names


def add_training_ops(class_count, inception_v3, learning_rate, final_tensor_name):
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


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # argmax on axis 1 returns indexes of max values in rows (single prediction)
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def evaluate_model_accuracy(sess, training_data, bottleneck_input, ground_truth_input, evaluation_step, prediction,
                            inception_v3):
    print('Evaluate accuracy of trained model on test data')
    test_bottlenecks, test_ground_truth, test_file_names = (
        get_all_bottlenecks(sess, training_data, 'testing', inception_v3)
    )
    test_accuracy, predictions = sess.run([evaluation_step, prediction],
                                          feed_dict={bottleneck_input: test_bottlenecks,
                                                     ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))
    print('Misclassified test images:')
    class_labels = list(training_data.keys())
    for i, test_filename in enumerate(test_file_names):
        expected_class_index = test_ground_truth[i].argmax()
        predicted_class_index = predictions[i]
        if predictions[i] != expected_class_index:
            predicted_class = class_labels[predicted_class_index]
            expected_class = class_labels[expected_class_index]
            print('%50s\t%8s != %8s' % (test_filename, predicted_class, expected_class))


def write_out_graph_and_labels(sess, graph, labels, final_tensor_name, output_dir='./output'):
    utils.ensure_dir_exists(output_dir)
    print('Write output to', output_dir)

    # write out trained weights
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [final_tensor_name])
    output_graph_file_name = os.path.join(output_dir, 'output_graph.pb')
    with tf.gfile.FastGFile(output_graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    # write out labels
    output_labels_file_name = os.path.join(output_dir, 'labels.txt')
    with tf.gfile.FastGFile(output_labels_file_name, 'w') as f:
        f.write('\n'.join(labels) + '\n')


def main():
    # training parameters
    training_steps = 16000
    batch_size = 1000
    validation_batch_size = 100
    learning_rate = 0.01
    final_tensor_name = 'retrained_result'

    # import Inception-v3 model
    inception_v3 = InceptionV3()
    training_data = utils.get_training_data()
    cache_bottlenecks(training_data, inception_v3)

    train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor = (
        add_training_ops(len(training_data), inception_v3, learning_rate, final_tensor_name)
    )
    evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to summaries dir
    summaries_dir = utils.create_summaries_dir()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', inception_v3.graph)
    validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')
    print('Run "tensorboard --logdir', summaries_dir, '" to visualize training process')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('Start training')
        for i in range(training_steps):
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                sess, training_data, batch_size, 'training', inception_v3)
            # Feed the bottlenecks and ground truth into the graph, and run a training step.
            # Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Validate model performance
            if (i % 10 == 0) or (i == training_steps + 1):
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy_mean],
                                                               feed_dict={bottleneck_input: train_bottlenecks,
                                                                          ground_truth_input: train_ground_truth})
                print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                validation_bottlenecks, validation_ground_truth, _ = (get_random_cached_bottlenecks(
                    sess, training_data, validation_batch_size, 'validation', inception_v3)
                )
                validation_summary, validation_accuracy = sess.run([merged, evaluation_step],
                                                                   feed_dict={bottleneck_input: validation_bottlenecks,
                                                                              ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (datetime.now(), i,
                                                                            validation_accuracy * 100,
                                                                            len(validation_bottlenecks)))

        evaluate_model_accuracy(sess, training_data, bottleneck_input, ground_truth_input, evaluation_step, prediction,
                                inception_v3)

        write_out_graph_and_labels(sess, inception_v3.graph, list(training_data.keys()), final_tensor_name)


if __name__ == '__main__':
    main()
