# -*- coding: utf-8 -*-
"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import logging

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# import main.data_utils as data_utils
import main.sequence_model as sequence_model
import argparse,pickle

import subprocess
import stat

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_layers", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("target_size", 36, "class number.")
tf.app.flags.DEFINE_integer("word_embedding_size", 339, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/argument_raw/argument_data_form.data", "Data directory")
tf.app.flags.DEFINE_string("saver_dir", "D:/Code/pycharm/Sequence-Label-Attention/main/saver/argument_saver/", "saver directory.")
tf.app.flags.DEFINE_boolean("use_attention", True, "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 60, "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.7, "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True, "Use birectional RNN")
tf.app.flags.DEFINE_integer("epoch", 50, "数据集共训练100次.")
FLAGS = tf.app.flags.FLAGS


def create_model(session):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = sequence_model.SequenceModel(
            FLAGS.max_sequence_length,
            FLAGS.target_size,
            FLAGS.word_embedding_size,
            FLAGS.hidden_layers,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            learning_rate=FLAGS.learning_rate,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn)
    with tf.variable_scope("model", reuse=True):
        model_test = sequence_model.SequenceModel(
            FLAGS.max_sequence_length,
            FLAGS.target_size,
            FLAGS.word_embedding_size,
            FLAGS.hidden_layers,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            learning_rate=FLAGS.learning_rate,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn)

    # ckpt = tf.train.get_checkpoint_state(FLAGS.saver_dir)
    # if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #     model_train.saver.restore(session, ckpt.model_checkpoint_path)
    # else:
    #     print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model_train, model_test


def get_max_from_list(pred_type,event_template):

    max_index=-1
    max_value=float('-Inf')
    for i in event_template:
        if pred_type[i]>max_value:
            max_value=pred_type[i]
            max_index=i

    return max_index


def argument_calculate_f_score(pred_test, tag_test, L_test,T_test, iter_num):

    # data_f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/temp_data.data", 'rb')
    # pred_test, tag_test, L_test, T_test = pickle.load(data_f)
    # data_f.close()

    template_f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/event_template.data", 'rb')
    event_template_num_dict,event_time_list = pickle.load(template_f)
    template_f.close()

    prediction=[]

    for i,(event_type,pred_test) in enumerate(zip(T_test,pred_test)):
        event_template=event_template_num_dict[event_type]
        event_template.append(0)
        sen_pred=[]
        for pred_type in pred_test:
            max_index=get_max_from_list(pred_type,event_template)
            sen_pred.append(max_index)

        prediction.append(sen_pred)


    #========================计算F值-===========================
    iden_p=0   # 识别的个体总数
    iden_r=0    # 测试集中存在个个体总数
    iden_acc=0  # 正确识别的个数

    classify_p = 0  # 识别的个体总数
    classify_r = 0  # 测试集中存在个个体总数
    classify_acc = 0  # 正确识别的个数

    for i in range(len(L_test)):
        for j in range(L_test[i]):
            if prediction[i][j]!=0:
                classify_p+=1
                iden_p+=1

            if tag_test[i][j]!=0:
                classify_r+=1
                iden_r+=1

            if (tag_test[i][j] in event_time_list and prediction[i][j] in event_time_list) or (tag_test[i][j]==prediction[i][j] and tag_test[i][j]!=0):
                classify_acc+=1

            if prediction[i][j]!=0 and tag_test[i][j]!=0:
                iden_acc+=1

    try:
        print('-----------------------' + str(iter_num) + '-----------------------------')
        print('Argument Identification:')
        print(str(iden_acc) + '------' + str(iden_p) + '------' + str(iden_r))
        p = iden_acc / iden_p
        r = iden_acc / iden_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
        print('Argument Classification:')
        print(str(classify_acc) + '------' + str(classify_p) + '------' + str(classify_r))
        p = classify_acc / classify_p
        r = classify_acc / classify_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
            print('-----------------------' + str(iter_num) + '-----------------------------')
            return f
    except ZeroDivisionError:
        print('all zero')
        return 0


def get_event_type_list(x):
    # 触发词类型
    T_test=[]
    X_test=np.array(x)
    X_test=X_test[:,:,305:]
    for i in X_test:
        for j in i:
            if 1.0 in j:
                T_test.append(j.tolist().index(1.0))
                break
    return T_test


def train():
    print('Applying Parameters:')
    for k, v in FLAGS.__dict__['__flags'].items():
        print('%s: %s' % (k, str(v)))
    print("Preparing data in %s" % FLAGS.data_dir)

    data_f = open(FLAGS.data_dir, 'rb')
    X_train, tag_train, L_train, Weights_train, W_train, X_test, tag_test, L_test, Weights_test, W_test, X_dev, tag_dev, L_dev, Weights_dev, W_dev = pickle.load(data_f)
    data_f.close()

    print(np.array(X_train).shape)
    print(np.array(tag_train).shape)
    print(np.array(L_train).shape)
    print(np.array(Weights_train).shape)
    print(np.array(W_train).shape)

    print(np.array(X_test).shape)
    print(np.array(tag_test).shape)
    print(np.array(L_test).shape)
    print(np.array(Weights_test).shape)
    print(np.array(W_test).shape)

    print(np.array(X_dev).shape)
    print(np.array(tag_dev).shape)
    print(np.array(L_dev).shape)
    print(np.array(Weights_dev).shape)
    print(np.array(W_dev).shape)
    # sys.exit()

    # 分配TensorFlow占用的显存
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:

        model, model_test = create_model(sess)

        max_f_score=0

        # # test 测试集计算
        # test_pred = sess.run(model_test.tagging_output, {model_test.input_data:X_test, model_test.tags: tag_test
        #             , model_test.tag_weights: Weights_test, model_test.sequence_length:L_test})
        # current_step_f_score=calculate_f_score(test_pred, tag_test, L_test, "test:")
        #
        # sys.exit()
        for e in range(FLAGS.epoch):
            for ptr in range(0, len(X_train), FLAGS.batch_size):
                X_batch_data=X_train[ptr:ptr + FLAGS.batch_size]
                tag_batch_data=tag_train[ptr:ptr + FLAGS.batch_size]
                L_batch_data=L_train[ptr:ptr + FLAGS.batch_size]
                Weights_batch_data=Weights_train[ptr:ptr + FLAGS.batch_size]
                W_batch_data=W_train[ptr:ptr + FLAGS.batch_size]

                sess.run(model.update, {model.input_data:X_batch_data, model.tags: tag_batch_data,
                                        model.tag_weights: Weights_batch_data, model.sequence_length:L_batch_data})

            if e % 10 == 0:
                test_num = 1000
                pred = sess.run(model.tagging_output, {model.input_data:X_train[:test_num], model.tags: tag_train[:test_num]
                    , model.tag_weights: Weights_train[:test_num], model.sequence_length:L_train[:test_num]})
                T_test=get_event_type_list(X_train)
                argument_calculate_f_score(pred,tag_train[:test_num],L_train[:test_num],T_test[:test_num], "train:"+str(e))

            # test 测试集计算
            test_pred = sess.run(model_test.tagging_output, {model_test.input_data:X_test, model_test.tags: tag_test
                    , model_test.tag_weights: Weights_test, model_test.sequence_length:L_test})

            T_test=get_event_type_list(X_test)
            current_step_f_score=argument_calculate_f_score(test_pred, tag_test, L_test,T_test, "test:"+str(e))
            if max_f_score<current_step_f_score:
                max_f_score=current_step_f_score
                if max_f_score > 0.5:

                    log_file_path = os.path.join(FLAGS.saver_dir+"model_"+str(max_f_score)+".log")
                    log_file=open(log_file_path, "w")
                    for k, v in FLAGS.__dict__['__flags'].items():
                        write_str=str('%s: %s' % (k, str(v)))+"\n"
                        log_file.write(write_str)
                    log_file.close()
                    checkpoint_path = os.path.join(FLAGS.saver_dir, "model_"+str(max_f_score)+".ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print("max_f_score: "+str(max_f_score))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
