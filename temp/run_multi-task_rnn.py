# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import temp.data_utils as data_utils
import temp.multi_task_model as multi_task_model

import subprocess
import stat

# tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 300, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "data/ATIS_samples", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "tmp/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 30000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", "tagging", "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    print('Please indicate task to run.' +
          'Available options: intent; tagging; joint')
    exit()

# task = dict({'intent': 0, 'tagging': 0, 'joint': 0})
# task['tagging'] = 1

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


# _buckets = [(3, 10), (10, 25)]

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '\conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(bytes("".join(open(filename, "r").readlines()), encoding="utf8"))
    for line in str(stdout, encoding="utf-8").split("\n"):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the word sequence.
      target_path: path to the file with token-ids for the tag sequence;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      label_path: path to the file with token-ids for the intent label
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target, label) tuple read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1];source, target, label are lists of token-ids
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(label_path, mode="r") as label_file:
                source = source_file.readline()
                target = target_file.readline()
                label = label_file.readline()
                counter = 0
                while source and target and label and (not max_size \
                                                               or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    label_ids = [int(x) for x in label.split()]
                    #          target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids, label_ids])
                            break
                    source = source_file.readline()
                    target = target_file.readline()
                    label = label_file.readline()
    return data_set  # 3 outputs in each unit: source_ids, target_ids, label_ids


def create_model(session,
                 source_vocab_size,
                 target_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size, FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn)
    with tf.variable_scope("model", reuse=True):
        model_test = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def train():
    print('Applying Parameters:')
    for k, v in FLAGS.__dict__['__flags'].items():
        print('%s: %s' % (k, str(v)))
    print("Preparing data in %s" % FLAGS.data_dir)

    date_set = data_utils.prepare_multi_task_data(FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

    in_seq_train, out_seq_train, label_train = date_set[0]
    in_seq_dev, out_seq_dev, label_dev = date_set[1]
    in_seq_test, out_seq_test, label_test = date_set[2]
    vocab_path, tag_vocab_path, _ = date_set[3]

    print(date_set)

    result_dir = FLAGS.train_dir + '/test_results'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    current_taging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
    current_taging_test_out_file = result_dir + '/tagging.test.hyp.txt'

    # 每个单词对应一个数字，vocab是单词对应数字的dict，rev_vocab是所有单词的list
    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)

    # print(vocab)
    # print(rev_vocab)
    # print(tag_vocab)
    # print(rev_tag_vocab)

    # 分配TensorFlow占用的显存
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # Create model.
        print("Max sequence length: %d." % _buckets[0][0])
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess, len(vocab), len(tag_vocab))
        print("Creating model with source_vocab_size=%d, target_vocab_size=%d." % (len(vocab), len(tag_vocab)))

        # Read data into buckets and compute their sizes.
        print("Reading train/valid/test data (training set limit: %d)."
              % FLAGS.max_train_data_size)

        dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
        test_set = read_data(in_seq_test, out_seq_test, label_test)
        train_set = read_data(in_seq_train, out_seq_train, label_train)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        print(train_set)
        print(test_set)

        # This is the training loop.
        current_step = 0

        best_valid_score = 0
        best_test_score = 0

        while model.global_step.eval() < FLAGS.max_training_steps:

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            batch_data = model.get_batch(train_set, bucket_id)
            # 获取输入数据，标注等
            encoder_inputs, tags, tag_weights, batch_sequence_length, _ = batch_data

            # loss损失函数值，
            _, step_loss, tagging_logits = model.tagging_step(sess, encoder_inputs, tags, tag_weights,
                                                              batch_sequence_length, bucket_id, False)

            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:

                def run_valid_test(data_set, mode):  # mode: Eval, Test
                    # Run evals on development/test set and print the accuracy.
                    word_list = list()
                    ref_tag_list = list()
                    hyp_tag_list = list()

                    for bucket_id in range(len(_buckets)):
                        for i in range(len(data_set[bucket_id])):
                            sample = model_test.get_one(data_set, bucket_id, i)
                            encoder_inputs, tags, tag_weights, sequence_length, _ = sample
                            # print("==================="+str(bucket_id)+"=================="+str(i)+"===========================")
                            # print("encoder_inputs")
                            # print(encoder_inputs)
                            # print("tags")
                            # print(tags)
                            # print("tag_weights")
                            # print(tag_weights)
                            # print("sequence_length")
                            # print(sequence_length)
                            # print("==================="+str(bucket_id)+"=================="+str(i)+"===========================")

                            _, step_loss, tagging_logits = model_test.tagging_step(sess, encoder_inputs, tags,
                                                                                   tag_weights, sequence_length,
                                                                                   bucket_id, True)

                            word_list.append([rev_vocab[x[0]] for x in encoder_inputs[:sequence_length[0]]])
                            ref_tag_list.append([rev_tag_vocab[x[0]] for x in tags[:sequence_length[0]]])
                            hyp_tag_list.append([rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:sequence_length[0]]])

                    taging_out_file = current_taging_test_out_file
                    tagging_eval_result = conlleval(hyp_tag_list, ref_tag_list, word_list, taging_out_file)
                    sys.stdout.flush()

                    return tagging_eval_result

                test_tagging_result = run_valid_test(test_set, 'Test')
                print(test_tagging_result)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
