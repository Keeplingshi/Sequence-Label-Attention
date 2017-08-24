# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:28:22 2016

@author: Bing Liu (liubing@cmu.edu)

Multi-task RNN model with an attention mechanism.
  - Developped on top of the Tensorflow seq2seq_model.py example: 
    https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
  - Note that this example code does not include output label dependency modeling.
    One may add a loop function as in the rnn_decoder function in tensorflow
    seq2seq.py example to feed emitted label embedding back to RNN state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# import temp.data_utils as data_utils
import main.seq_labeling as seq_labeling
# import temp.seq_classification as seq_classification
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn
import sys


class MultiTaskModel(object):
    def __init__(self,
                 sentence_length,   # 补零截取后的句子长度，固定值，60
                 tag_size,    # 类别个数 class_num
                 word_embedding_size,      # 词向量长度 300
                 cell_size,     # 隐藏层数目 128
                 num_layers,    # rnn层数
                 max_gradient_norm, # clip_by_global_norm参数
                 batch_size,    # 每次训练多少条
                 dropout_keep_prob=1.0,     # dropout
                 learning_rate=0.001,
                 bidirectional_rnn=True,
                 use_attention=False,
                 forward_only=False):
        self.tag_size = tag_size
        self.word_embedding_size = word_embedding_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional_rnn = bidirectional_rnn
        self.global_step = tf.Variable(0, trainable=False)

        # 输入数据、标签、标签权重
        self.input_data = tf.placeholder(tf.float32, [None, sentence_length, word_embedding_size], name="input_data")
        self.tags = tf.placeholder(tf.float32, [None, sentence_length], name="tags")
        self.tag_weights = tf.placeholder(tf.float32, [None, sentence_length], name="tag_weights")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        self.tr_input_data=tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2]))
        self.tr_tags=tf.transpose(self.tags, perm=[1, 0])
        self.tr_tag_weights=tf.transpose(self.tag_weights, perm=[1, 0])

        # If we use sampled softmax, we need an output projection.
        softmax_loss_function = None

        # Create the internal multi-layer cell for our RNN.
        def create_cell():
            if not forward_only and dropout_keep_prob < 1.0:
                single_cell = lambda: BasicLSTMCell(self.cell_size)
                cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
                cell = DropoutWrapper(cell,
                                      input_keep_prob=dropout_keep_prob,
                                      output_keep_prob=dropout_keep_prob)
            else:
                single_cell = lambda: BasicLSTMCell(self.cell_size)
                cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
            return cell

        # 创建rnn单元
        self.cell_fw = create_cell()
        self.cell_bw = create_cell()

        # rnn的输出
        base_rnn_output = self.generate_rnn_output()
        encoder_outputs, encoder_state, attention_states = base_rnn_output

        # 求 预测值和loss损失函数的过程
        seq_labeling_outputs = seq_labeling.generate_sequence_output(encoder_outputs,
                                                                     encoder_state,
                                                                     self.tr_tags,
                                                                     self.sequence_length,
                                                                     self.tag_size,
                                                                     self.tr_tag_weights,
                                                                     softmax_loss_function=softmax_loss_function,
                                                                     use_attention=use_attention)
        self.tagging_output, self.tagging_loss = seq_labeling_outputs
        # self.tagging_output=tf.transpose(tf.squeeze(self.tagging_output), perm=[1, 0, 2])
        self.tagging_output=tf.transpose(self.tagging_output, perm=[1, 0, 2])
        self.loss = self.tagging_loss

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdamOptimizer(learning_rate)
            gradients = tf.gradients(self.tagging_loss, params)

            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def generate_rnn_output(self):
        """
        Generate RNN state outputs with word embeddings as inputs
        """
        with tf.variable_scope("generate_seq_output"):
            # encoder_emb_inputs是输入数据  数据格式和trigger_rnn一样  [sequence_length, batch_size, word_embedding_size]
            rnn_outputs = static_bidirectional_rnn(self.cell_fw, self.cell_bw, self.tr_input_data, sequence_length=self.sequence_length, dtype=tf.float32)
            encoder_outputs, encoder_state_fw, encoder_state_bw = rnn_outputs
            # with state_is_tuple = True, if num_layers > 1,
            # here we simply use the state from last layer as the encoder state
            state_fw = encoder_state_fw[-1]
            state_bw = encoder_state_bw[-1]
            encoder_state = tf.concat([tf.concat(state_fw, 1), tf.concat(state_bw, 1)], 1)
            top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size + self.cell_bw.output_size]) for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)

            return encoder_outputs, encoder_state, attention_states


    # # session.run
    # def tagging_step(self, session, encoder_inputs, tags, tag_weights,
    #                  batch_sequence_length, bucket_id, forward_only):
    #     """Run a step of the tagging model feeding the given inputs.
    #
    #     Args:
    #       session: tensorflow session to use.
    #       encoder_inputs: list of numpy int vectors to feed as encoder inputs.
    #       tags: list of numpy int vectors to feed as decoder inputs.
    #       tag_weights: list of numpy float vectors to feed as target weights.
    #       batch_sequence_length: batch_sequence_length
    #       bucket_id: which bucket of the model to use.
    #       forward_only: whether to do the backward step or only forward.
    #
    #     Returns:
    #       A triple consisting of gradient norm (or None if we did not do backward),
    #       average perplexity, and the output tags.
    #
    #     Raises:
    #       ValueError: if length of encoder_inputs, decoder_inputs, or
    #         target_weights disagrees with bucket size for the specified bucket_id.
    #     """
    #     # Check if the sizes match.
    #
    #     print("===============1=====================")
    #     print(tags)
    #     print(np.array(tags).shape)
    #
    #     encoder_size, tag_size = self.buckets[bucket_id]
    #     if len(encoder_inputs) != encoder_size:
    #         raise ValueError("Encoder length must be equal to the one in bucket,"
    #                          " %d != %d." % (len(encoder_inputs), encoder_size))
    #     if len(tags) != tag_size:
    #         raise ValueError("Decoder length must be equal to the one in bucket,"
    #                          " %d != %d." % (len(tags), tag_size))
    #
    #     # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    #     input_feed = {}
    #     input_feed[self.sequence_length.name] = batch_sequence_length
    #     for l in range(encoder_size):
    #         input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    #         input_feed[self.tags[l].name] = tags[l]
    #         input_feed[self.tag_weights[l].name] = tag_weights[l]
    #
    #     # Output feed: depends on whether we do a backward step or not.
    #     if not forward_only:
    #         output_feed = [self.update,  # Update Op that does SGD.
    #                        self.gradient_norm,  # Gradient norm.
    #                        self.loss]  # Loss for this batch.
    #         for i in range(tag_size):
    #             output_feed.append(self.tagging_output[i])
    #     else:
    #         output_feed = [self.loss]
    #         for i in range(tag_size):
    #             output_feed.append(self.tagging_output[i])
    #
    #     outputs = session.run(output_feed, input_feed)
    #     if not forward_only:
    #         return outputs[1], outputs[2], outputs[3:3 + tag_size]
    #     else:
    #         return None, outputs[0], outputs[1:1 + tag_size]


