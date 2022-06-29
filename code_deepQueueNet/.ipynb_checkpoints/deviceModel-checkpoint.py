# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu
# we learned from 'parse' and 'tfrecord_input_fn' functions, but adapted to our case with our domain knowledge

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache-2.0 License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache-2.0 License for more details.
import tensorflow as tf
import numpy as np
import pandas as pd
from code_deepQueueNet.featureET import trace2Samples, buildfolder
import glob
import os, shutil
import time
import numba
import warnings
warnings.filterwarnings("ignore")




"""
This script is to implement the deepQueueNet in tensorflow 1.13.1.   
     1. tfrecords_count       count no. of tfrecords in a file
     2. _loop_size            return: batch_per_epoch
     3. parse                 parse tfrecords
     4. tfrecords_input_fn    minibatch generator 
     5. _create_one_cell      building cell for lstm 
     6. build_and_training    construct and run device model
"""

 


class deepQueueNet(trace2Samples):

    def __init__(self, config, target, data_preprocessing=False):
        super(deepQueueNet, self).__init__(config, target)

        if data_preprocessing:
            buildfolder(config.modelname)
            self.fet_input()
            self._2hdf()
            self._min_max()
            self.model_input()
            self._loop_size()
            self.merge_sample(task='train')
            self.merge_sample(task='test1')
            self.merge_sample(task='test2')
        else:
            self.load_scaler()
            self._loop_size()

    def tfrecords_count(self, file):
        return sum(1 for _ in tf.python_io.tf_record_iterator(file))

    def _loop_size(self):

        @numba.jit
        def loop():
            count = 0
            for file in file_list:
                count += self.tfrecords_count(file)
            return count

        file_list = glob.glob('./data/{}/_tfrecords/*.tfrecords'.format(
            self.config.modelname))
        self.batch_per_epoch = loop() // self.config.BATCH_SIZE

    def parse(self, serialized):
        with tf.device("/cpu:0"):
            with tf.name_scope('parse'):
                features = tf.parse_single_example(
                    serialized,
                    features={
                        'fet': tf.VarLenFeature(tf.float32),
                        'label': tf.VarLenFeature(tf.float32)
                    })
                for k in ['fet', 'label']:
                    features[k] = tf.sparse_tensor_to_dense(features[k])
        return {'fet': features['fet']}, features['label']

    def tfrecords_input_fn(self, shuffle_buf=1000000):
        """return a generator to fetch a batch size dataset from the training database"""
        file_list = glob.glob('./data/{}/_tfrecords/*.tfrecords'.format(
            self.config.modelname))
        files = tf.data.Dataset.from_tensor_slices(file_list)
        files = files.shuffle(len(file_list))
        ds = files.apply(
            tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
                                                cycle_length=4))

        if shuffle_buf:
            ds = ds.apply(
                tf.contrib.data.shuffle_and_repeat(shuffle_buf,
                                                   count=self.config.epochs))
        ds = ds.map(lambda buf: self.parse(buf), num_parallel_calls=2)

        shapes = ({'fet': [self.config.TIME_STEPS * len(self.fet_cols)]}, [1])
        ds = ds.padded_batch(self.config.BATCH_SIZE, shapes)
        ds = ds.prefetch(1)

        X, y = ds.make_one_shot_iterator().get_next()
        X = tf.reshape(X['fet'],
                       [-1, self.config.TIME_STEPS,
                        len(self.fet_cols)])
        return X, y

    def _create_one_cell(self, layer):
        """lstm: nn layer"""
        cell = tf.contrib.rnn.LSTMCell(
            self.config.lstm_params['cell_neurons'][layer],
            forget_bias=0,
            state_is_tuple=True)
        if self.config.lstm_params['keep_prob'] == 1.0:
            return cell
        else:
            return tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.config.lstm_params['keep_prob'])

    def build_and_training(self):
        """build the model"""
        tf.reset_default_graph()
        lstm_graph = tf.Graph()
        with lstm_graph.as_default():
            X, y = self.tfrecords_input_fn()
            tf.add_to_collection('X', X)

            #encoder
            #BLSTM is more accuracy than LSTM, but will also take more time to train.
            with tf.variable_scope('lstm1',
                                   initializer=tf.orthogonal_initializer(),
                                   reuse=None):
                multi_cell = tf.contrib.rnn.MultiRNNCell([
                    self._create_one_cell(_)
                    for _ in range(self.config.lstm_params['layer'])
                ],
                                                         state_is_tuple=True)
                val, _ = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
#                 multi_cell_bw=tf.contrib.rnn.MultiRNNCell([self._create_one_cell(_)
#                                                            for _ in range(self.config.lstm_params['layer'])],
#                                                           state_is_tuple=True)
#                 val, _=tf.nn.bidirectional_dynamic_rnn(multi_cell, multi_cell_bw, X, dtype=tf.float32)
#                 val = tf.concat(val, 2)
#                 val = tf.split(val, 2, -1)
#                 val= val[0] + val[1]

#attention layer
            initializer = tf.contrib.layers.variance_scaling_initializer(
                mode='FAN_AVG')
            rnn1 = tf.reshape(val, [
                -1, self.config.TIME_STEPS,
                self.config.lstm_params['cell_neurons'][-1]
            ])
            for mul_head in range(self.config.mul_head):
                with tf.variable_scope('head{}'.format(mul_head + 1),
                                       initializer=tf.orthogonal_initializer(),
                                       reuse=None):
                    V = tf.layers.dense(inputs=rnn1,
                                        units=self.config.att,
                                        kernel_initializer=initializer)
                    K = tf.layers.dense(inputs=rnn1,
                                        units=self.config.att,
                                        kernel_initializer=initializer)
                    Q = tf.layers.dense(inputs=rnn1,
                                        units=self.config.att,
                                        kernel_initializer=initializer)
                    score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                        self.config.att)
                    softmax = tf.nn.softmax(score, dim=2)
                    z = tf.matmul(softmax, V)
                    if mul_head == 0:
                        Z = z
                    else:
                        Z = tf.concat([Z, z], -1)
            O = tf.layers.dense(inputs=Z,
                                units=self.config.mul_head_output_nodes,
                                kernel_initializer=initializer)
            dense_outputs = tf.reshape(O, [
                -1, self.config.TIME_STEPS, self.config.mul_head_output_nodes
            ])

            #decoder layer
            with tf.variable_scope('lstm2',
                                   initializer=tf.orthogonal_initializer(),
                                   reuse=None):
                multi_cell2 = tf.contrib.rnn.MultiRNNCell([
                    self._create_one_cell(_)
                    for _ in range(self.config.lstm_params['layer'])
                ],
                                                          state_is_tuple=True)
                val2, __ = tf.nn.dynamic_rnn(multi_cell2,
                                             dense_outputs,
                                             dtype=tf.float32)


#                 multi_cell2_bw=tf.contrib.rnn.MultiRNNCell([self._create_one_cell(_)
#                                                             for _ in range(self.config.lstm_params['layer'])],
#                                                            state_is_tuple=True)
#                 val2, __=tf.nn.bidirectional_dynamic_rnn(multi_cell2, multi_cell2_bw, dense_outputs, dtype=tf.float32)
#                 val2 = tf.concat(val2, 2)
#                 val2 = tf.split(val2, 2, -1)
#                 val2= val2[0] + val2[1]

#output layer
            rnn2 = tf.reshape(
                val2, [-1, self.config.lstm_params['cell_neurons'][-1]])
            stacked_outputs = tf.layers.dense(
                rnn2,
                self.config.n_outputs,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    self.config.l2),
                kernel_initializer=initializer)
            outputs = tf.reshape(
                stacked_outputs,
                [-1, self.config.TIME_STEPS, self.config.n_outputs])
            outputs = outputs[:, self.config.TIME_STEPS -
                              1, :]  #keep only the last output of the seq
            tf.add_to_collection('outputs', outputs)

            #loss
            loss = tf.reduce_mean(tf.square(outputs - y))
            #regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(loss)
        """train the model"""
        mat_t = "    {:8}\t\t  {:32}   {:16}"
        mat_c = "    {:^16}\t{:^30}\t{:^18}\n"
        with tf.Session(graph=lstm_graph) as sess:
            path = './save/{}'.format(self.config.modelname)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            saver = tf.train.Saver(max_to_keep=5)
            f = open(path + '/acc.txt', 'w')
            max_mse = np.inf

            #Eval.
            self.mse_train_record = []
            self.mse_test1_record = []
            self.mse_test2_record = []
            self.x_train, y_train = self.load_sample('train')
            self.x_test1, y_test1 = self.load_sample('test1')
            self.x_test2, y_test2 = self.load_sample('test2')

            np.random.seed(self.config.seed)
            sess.run(tf.global_variables_initializer())
            it = 1
            t0 = time.time()
            print('Batch per epoch: {}\n'.format(self.batch_per_epoch))
            print(
                mat_c.format("iteration (in k)", "MSE- train/test1/test2",
                             "time used"))
            while True:
                try:
                    _, loss_value = sess.run([train_op, loss])
                    if it % 1000 == 0:
                        mse_test1 = loss.eval(feed_dict={
                            X: self.x_test1,
                            y: y_test1
                        })
                        mse_test2 = loss.eval(feed_dict={
                            X: self.x_test2,
                            y: y_test2
                        })
                        mse_train = loss.eval(feed_dict={
                            X: self.x_train,
                            y: y_train
                        })
                        f.write(str(it//1000)+'k, train_mse: '+str(mse_train)+\
                                ', test1_mse: '+str(mse_test1)+', test2_mse: '+str(mse_test2)+'\n')
                        if mse_test1 + mse_test2 < max_mse:
                            max_mse = mse_test1 + mse_test2
                            saver.save(sess,
                                       './save/{}/model.ckpt'.format(
                                           self.config.modelname),
                                       global_step=it // 1000)
                        print(
                            mat_t.format(
                                it // 1000,
                                "%f/%f/%f" % (mse_train, mse_test1, mse_test2),
                                "%.2f min." % ((time.time() - t0) / 60)))
                        self.mse_test1_record.append(mse_test1)
                        self.mse_test2_record.append(mse_test2)
                        self.mse_train_record.append(mse_train)
                        t0 = time.time()
                    it += 1
                except:
                    break
            f.close()
