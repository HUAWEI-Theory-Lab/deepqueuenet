"""
This script is to implement the following functions.   
     1. fet_input             cleaned dataset with useful columns only (.csv)
     2. _2hdf                 raw trace in .csv -> timeseries batches -> .hdf 
     3. _min_max              return data scaler from the training dataset 
     4. model_input          func: 
                                 - tfrecords for training 
                                 - hdf for evaluation 
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from code_deepQueueNet.tools.MinMaxScaler import cScaler, load_scaler
from code_deepQueueNet.tools import builtseq
from code_deepQueueNet.tools.trafficindicator import feaTure  #to create traffic indicators
from sklearn.model_selection import train_test_split
import glob
import h5py
import os, shutil
from multiprocessing import Process, Manager
import numba
import warnings
warnings.filterwarnings("ignore")






def buildfolder(root):
    for name in [
            '_hdf', '_hdf/train', '_hdf/test1', '_hdf/test2', '_scaler',
            '_tfrecords', '_error'
    ]:
        folder = './data/{}/{}'.format(root, name)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


class trace2Samples:

    def __init__(self, config, target):
        self.config = config
        self.target = target

    def multi_process(self, func, FILES, args=None):
        it = 0
        while True:
            files = FILES[it * self.config.no_process:(it + 1) *
                          self.config.no_process]
            if len(files) > 0:
                threads = []
                for file in files:
                    ARGS = list(args)
                    ARGS.append(file)
                    t = Process(target=func, args=tuple(ARGS))
                    threads.append(t)
                    t.start()
                for thr in threads:
                    thr.join()
                it += 1
            else:
                break

    def fet_input(self):
        """feature extraction module"""

        @numba.jit
        def gettraffic(dst_folder, my_fet, file):
            df = pd.read_csv(file).fillna(self.config.sp_wgt)

            #traffic load features
            ins = feaTure(df, self.config.no_of_port, self.config.no_of_buffer,
                          self.config.window, self.config.ser_rate)
            C_dst_SET, LOAD = ins.getCount()
            for i in range(self.config.no_of_port):
                df['TI%i' % i] = LOAD[i]
            for i in range(self.config.no_of_port):
                for j in range(self.config.no_of_buffer):
                    df['load_dst{}_{}'.format(i, j)] = C_dst_SET[(i, j)]

            #arrival patterns
            df['inter_arr_sys'] = df['timestamp (sec)'].diff()
            if self.config.no_of_buffer > 1:
                for i in range(self.config.no_of_buffer):
                    t = df[df['priority'] ==
                           i]['timestamp (sec)'].diff().rename(
                               'inter_arr{}'.format(i)).to_frame()
                    df = df.join(t)

            #save
            filename = file.split('/')[-1]
            drop_cols = ['timestamp (sec)'] + self.target
            if self.config.no_of_buffer == 1: drop_cols += ['priority']
            fet_cols = list(df.columns.drop(drop_cols))
            my_fet['fet_cols'] = fet_cols
            df[fet_cols + self.target].fillna(method='ffill').dropna().to_csv(
                '{}/{}'.format(dst_folder, filename), index=False)

        with Manager() as MG:
            my_fet = MG.dict()
            for mode in ['train', 'test']:
                file_dir = './data/{}/_traces/_{}'.format(
                    self.config.modelname, mode)
                FILES = []
                for dirpath, dirnames, filenames in os.walk(file_dir):
                    for file in filenames:
                        if (os.path.splitext(file)[1]
                                == '.csv') and 'checkpoint' not in file:
                            FILES.append(os.path.join(dirpath, file))

                dst_folder = './data/{}/_traces/{}'.format(
                    self.config.modelname, mode)
                if os.path.exists(dst_folder):
                    shutil.rmtree(dst_folder)
                os.makedirs(dst_folder)
                self.multi_process(gettraffic,
                                   FILES,
                                   args=(dst_folder, my_fet))
            self.fet_cols = my_fet['fet_cols']

    def load_hdf(self, file):
        with h5py.File(file, 'r') as hdf:
            x = hdf['x'][:]
            y = hdf['y'][:]
        return x, y

    def write_hdf(self, file, x, y):
        with h5py.File(file, 'w') as hdf:
            hdf['x'] = x
            hdf['y'] = y

    def write_hdf2(self, h, key, x, y):
        h['{}_x'.format(key)] = x
        h['{}_y'.format(key)] = y

    def _2hdf(self):
        """
        Build timeseries batches for bLSTM and save them in .hdf files
          - split train mode files for train and in-sample testing (test1);
          - save test mode files for out-of-sample testing (test2).
        """

        @numba.jit
        def split_2hdf(mode, file):
            key = file.split('/')[-1].split('.csv')[0]
            t = pd.read_csv(file)
            os.remove(file)

            ins = builtseq.build_timeseries(t.values, target_col_index=[-1])
            x, y = ins.timeseries(self.config.TIME_STEPS)
            "randomly selected part of the them. to represent the config"
            loc = np.random.choice(
                np.arange(len(y)),
                np.max([np.min([14000, len(y)]),
                        int(len(y) * 0.15)]),
                replace=False)
            x = x[loc]
            y = y[loc]
            if mode == 'train':
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=self.config.test_size, shuffle=True)
                self.write_hdf(
                    './data/{}/_hdf/train/{}.h5'.format(
                        self.config.modelname, key), x_train, y_train)
                self.write_hdf(
                    './data/{}/_hdf/test1/{}.h5'.format(
                        self.config.modelname, key), x_test, y_test)
            else:
                self.write_hdf(
                    './data/{}/_hdf/test2/{}.h5'.format(
                        self.config.modelname, key), x, y)

        for mode in ['train', 'test']:
            folder = './data/{}/_traces/{}'.format(self.config.modelname, mode)
            FILES = glob.glob('{}/*.csv'.format(folder))
            self.multi_process(split_2hdf, FILES, args=(mode, ))
            shutil.rmtree(folder)

    def _min_max(self):
        """cal. data_range from the training dataset."""
        src = './data/{}/_hdf/train/*.h5'.format(self.config.modelname)
        for i, file in enumerate(glob.glob(src)):
            x, y = self.load_hdf(file)
            xmin = pd.Series(x.min(axis=0).min(axis=0), index=self.fet_cols)
            xmax = pd.Series(x.max(axis=0).max(axis=0), index=self.fet_cols)
            ymin = pd.Series(np.array([y.min(axis=0).min(axis=0)]).flatten(),
                             index=self.target)
            ymax = pd.Series(np.array([y.max(axis=0).max(axis=0)]).flatten(),
                             index=self.target)
            if i == 0:
                self.x_MIN = xmin
                self.x_MAX = xmax
                self.y_MIN = ymin
                self.y_MAX = ymax
            else:
                self.x_MIN = np.minimum(self.x_MIN, xmin)
                self.x_MAX = np.maximum(self.x_MAX, xmax)
                self.y_MIN = np.minimum(self.y_MIN, ymin)
                self.y_MAX = np.maximum(self.y_MAX, ymax)

        ins = cScaler(self.x_MIN, self.x_MAX, 'x', self.fet_cols,
                      self.config.no_of_port, self.config.no_of_buffer)
        ins.cluster()
        ins.save('./data/{}/_scaler'.format(self.config.modelname))
        cScaler(self.y_MIN, self.y_MAX,
                'y').save('./data/{}/_scaler'.format(self.config.modelname))

    def load_scaler(self):
        folder = './data/{}/_scaler'.format(self.config.modelname)
        self.x_MIN, self.x_MAX, self.y_MIN, self.y_MAX, self.fet_cols, self.target = load_scaler(
            folder)

    def model_input(self):
        """timeseries batch in .hdf -> normalization -> save it in .tfrecords for training 
           -> sampling part of them for Eval. and save them in .hdf
        """

        @numba.jit
        def inner_op(mode, file, h):
            #load timeseries batch in .hdf
            key = file.split('/')[-1].split('.h5')[0]
            x, y = self.load_hdf(file)
            os.remove(file)

            #normalization
            x = (x - self.x_MIN.values) / (self.x_MAX.values -
                                           self.x_MIN.values)
            y = (y - self.y_MIN.values) / (self.y_MAX.values -
                                           self.y_MIN.values)

            #save it in .tfrecords
            if mode == 'train':
                writer = tf.python_io.TFRecordWriter(
                    './data/{}/_tfrecords/{}.tfrecords'.format(
                        self.config.modelname, key))
                """sampling part of them for eval. during the training phase."""
                loc = np.random.choice(np.arange(len(y)),
                                       int(len(y) * self.config.sub_rt),
                                       replace=False)
                x_valid = x[loc]
                y_valid = y[loc]
                self.write_hdf2(h, key, x_valid, y_valid)
                for a, d in zip(x, y):
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'fet':
                            tf.train.Feature(float_list=tf.train.FloatList(
                                value=a.reshape(1, -1)[0])),
                            'label':
                            tf.train.Feature(float_list=tf.train.FloatList(
                                value=d))
                        }))
                    writer.write(example.SerializeToString())
                writer.close()
            elif mode == 'test1':
                ratio = (1. - self.config.test_size) / self.config.test_size
                loc = np.random.choice(
                    np.arange(len(y)),
                    int(len(y) * self.config.sub_rt * ratio),
                    replace=False)
                x = x[loc]
                y = y[loc]
                self.write_hdf2(h, key, x, y)
            else:
                self.write_hdf2(h, key, x, y)

        folder = './data/{}/_hdf'.format(self.config.modelname)
        for mode in ['train', 'test1', 'test2']:
            src = '{}/{}/*.h5'.format(folder, mode)
            with h5py.File('{}/{}.h5'.format(folder, mode), mode='w') as h:
                for file in glob.glob(src):
                    inner_op(mode, file, h)
                shutil.rmtree('{}/{}'.format(folder, mode))

    def merge_sample(self, task):
        """merge all of the Eval. samples for further use."""
        with h5py.File(
                './data/{}/_hdf/{}.h5'.format(self.config.modelname, task),
                'r+') as hdf:
            for i, key in enumerate(
                [key[:-2] for key in hdf.keys() if '_x' in key]):
                if i == 0:
                    x = hdf['{}_x'.format(key)][:]
                    y = hdf['{}_y'.format(key)][:]
                else:
                    x = np.concatenate((x, hdf['{}_x'.format(key)][:]), axis=0)
                    y = np.concatenate((y, hdf['{}_y'.format(key)][:]), axis=0)
            hdf['x'] = x
            hdf['y'] = y

    def load_sample(self, task, saved_sample=None):
        if saved_sample:
            file = saved_sample
        else:
            file = './data/{}/_hdf/{}.h5'.format(self.config.modelname, task)
        with h5py.File(file, 'r') as hdf:
            x = hdf['x'][:]
            y = hdf['y'][:]
        return x, y
