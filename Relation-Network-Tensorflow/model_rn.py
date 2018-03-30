from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    print("Please check tfplot!!!")
    pass

from ops import conv2d, fc
from util import log, check_tensor

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                # check_tensor(o_i, name="o_i")  [batch, 26]
                # check_tensor(o_j, name="o_j")  [batch, 26]
                # check_tensor(q, name='q')      [batch, 11]
                g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
                g_2 = fc(g_1, 256, name='g_2')
                g_3 = fc(g_2, 256, name='g_3')
                g_4 = fc(g_3, 256, name='g_4')
                return g_4

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, q, scope='CONV'):
            with tf.variable_scope(scope) as scope:

                log.warn(scope.name)
                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')

                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                d = conv_4.get_shape().as_list()[1]
                all_g, all_feature, all_question = [], [], []

                for i in range(d*d):
                    o_i = conv_4[:, int(i / d), int(i % d), :]
                    o_i = concat_coor(o_i, i, d)
                    for j in range(d*d):
                        o_j = conv_4[:, int(j / d), int(j % d), :]
                        o_j = concat_coor(o_j, j, d)

                        # ================================================================
                        all_feature.append(tf.concat([o_i, o_j], axis=1))
                        all_question.append(q)
                        # ================================================================

                        if i == 0 and j == 0:
                            g_i_j = g_theta(o_i, o_j, q, reuse=False)
                        else:
                            g_i_j = g_theta(o_i, o_j, q, reuse=True)
                        all_g.append(g_i_j)

                all_g = tf.stack(all_g, axis=0)

                # ====================================================================================================
                # Added for weights before the first MLP.
                all_question = tf.stack(all_question, axis=0)
                all_feature = tf.stack(all_feature, axis=0)

                q_len = all_question.get_shape().as_list()[2]
                all_question = tf.stack(all_question, axis=0)
                all_feature = tf.stack(all_feature, axis=0)

                converted_feature = tf.nn.tanh(tl.fully_connected(\
                    all_feature, q_len, reuse=tf.AUTO_REUSE, scope='convert_fc'), name='convert_tanh')

                # weights1, weights2 [d*d, batch, 1]
                # all_question [d*d, batch, 11]

                weights_1 = get_weights(all_question, converted_feature)
                all_question_1 = tf.add(all_question, \
                                       tf.multiply(weights_1, all_question, name="step1_mul"), name="step1_add")
                weights_2 = get_weights(all_question_1, converted_feature)
                all_g = tf.multiply(all_g, weights_2, name="weighting_all_g")

                # new_question = tf.add(all_question, \
                #                        tf.multiply(weights_2, all_question_1, name="final_mul"), name="final_add")

                # check_tensor(all_g, "all_g") [d*d, batch, concated_feature_dim]
                # final_question [d*d, batch, 11], should merge into all_g

                # old_len = all_g.get_shape().as_list()[2]
                # features, _ = tf.split(all_g, [old_len - q_len,q_len], axis=2, name="split_old")
                # all_g = tf.concat([features, new_question], axis=2)

                # ====================================================================================================

                all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
                return all_g

        def get_weights(all_q, all_f, scope = 'WEIGHTS'):
            # all_q, all_f[d*d, batch, 11]
            # weight [d*d, batch, 1]
            # check_tensor(all_q)
            # check_tensor(all_f)
            # print("=======")

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                q_len = all_q.get_shape().as_list()[2]
                h = tf.nn.tanh(tf.add(tl.fully_connected(all_f, q_len, biases_initializer=None, scope="IA_fc"),\
                                      tl.fully_connected(all_q, q_len, scope="QA_fc"),\
                                      name='weight_add'), name='weight_tanh')

                weight = tf.nn.softmax(\
                    tl.fully_connected(h, 1, scope="to_weight"), axis=0, name="weight_softmax")
            check_tensor(weight, name='weight')
            return weight


        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, n, activation_fn=None, name='fc_3')
                return fc_3

        g = CONV(self.img, self.q, scope='CONV')
        logits = f_phi(g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)

        # Add summaries
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=4,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
