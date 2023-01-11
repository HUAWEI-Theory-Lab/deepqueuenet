
import os, sys, time
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset

import copy
import numba
import pandas as pd
from sklearn.model_selection import train_test_split

from multiprocessing import Process, Manager
import glob
import h5py
import warnings

warnings.filterwarnings("ignore")

from tf_impl.trafficindicator import feaTure
from tf_impl.buildseq import build_timeseries
from tf_impl.minmaxscaler import cScaler, load_scaler

from config import BaseConfig



class DeviceDataset(Dataset):
    def __init__(self, input_file: list, seq_len : int, use_gpu: bool = True, input_dim : int = None, verbose = False, item_to_cuda : bool = False):
        
        self.data = []
        self.data_len = []
        self.seq_len = seq_len
        self.input_file = input_file
        self.item_to_cuda = item_to_cuda

        self.input_dim = input_dim if input_dim is not None else -1 # -1 means all cols 
        
        input_file_len = len(input_file)
        for file_idx, file_name in enumerate(input_file):
            print("\r", end="")
            print("Data load: {} / {}".format(file_idx+1 , input_file_len), end = "")
            try:
                with open(file_name, "rb") as f:
                    one_data = np.load(f)
                    self.data.append(one_data)
                    self.data_len.append(len(one_data) - (self.seq_len - 1))
            except:
                raise FileNotFoundError(
                    "\"{}\" data file not found".format(file_name))
        print()
        
        assert len(self.data) > 0
        for data_item in self.data:
            assert len(data_item) >= self.seq_len
        
        self.all_data_len = sum(self.data_len)
        # self.data = np.array(self.data)
        
        if verbose:
            print("Dataset loader ({})".format(
                    list(
                        zip(input_file, self.data_len)
                        )
                ))
        
        self.data_start_idx = [0]
        for idx in range(1, len(self.data)):
            self.data_start_idx.append(self.data_len[idx-1] + self.data_start_idx[idx-1])
        
        for idx in range(len(self.data)):
            self.data[idx] = torch.from_numpy(self.data[idx]).float()

        if use_gpu:
            try:
                for idx in range(len(self.data)):
                    self.data[idx] = self.data[idx].cuda()
            except:
                raise Exception("data to cuda failed!")

    def __len__(self):
        return self.all_data_len
    
    def get_data_len(self):
        return copy.deepcopy(self.data_len)
    
    def get_all_data_col(self, col):
        data_list = []
        for data_item in self.data:
            data_list.append(data_item[self.seq_len-1:, col])
        data_list = np.concatenate(data_list)
        return data_list
    
    def normalize_data(self, col_max_min_dict):
        def max_min_normalize(col, col_max, col_min, epsilon = 1e-16):
            return ((col - col_min) / (col_max - col_min + epsilon))
        
        data_num = len(self.data)
        for idx, data_item in enumerate(self.data):
            print("\r", end="")
            print("Dataset normalize {} / {}".format(idx + 1, data_num), end="")
            for col_l in col_max_min_dict.keys():
                col_max = col_max_min_dict[col_l][0]
                col_min = col_max_min_dict[col_l][1]
                for col in col_l:
                    self.data[idx][:,col] = max_min_normalize(data_item[:,col], col_max = col_max, col_min = col_min)
        print()
    
    def test_run(self, file_idx = 0):
        with open(self.input_file[file_idx], "rb") as f:
            one_data = np.load(f)
            assert self.input_dim + 1 == len(one_data[0]), f"Input dim value: {self.input_dim}, data feature num: {len(one_data[0])}"

    def __getitem__(self, idx_in: int):
        for data_idx, (start_idx, one_data_len) in enumerate(zip(self.data_start_idx, self.data_len)):
            if (idx_in >= start_idx) and (idx_in < start_idx + one_data_len):
                idx = idx_in -  start_idx
                if self.item_to_cuda:
                    return self.data[data_idx][idx: idx + self.seq_len, : self.input_dim].cuda(), self.data[data_idx][idx + self.seq_len - 1, -1].cuda()
                else:
                    return self.data[data_idx][idx: idx + self.seq_len, : self.input_dim], self.data[data_idx][idx + self.seq_len - 1, -1]

        raise Exception("Data not fetched for index {}!".format(idx_in))





# @numba.jit
def process_trace(file, base_dir, mode):
    config = BaseConfig()
    target_col = ['time_in_sys']

    df = pd.read_csv(file).fillna(config.sp_wgt)

    #traffic load features
    ins = feaTure(df, config.no_of_port, config.no_of_buffer,
                    config.window, config.ser_rate)
    C_dst_SET, LOAD = ins.getCount()
    for i in range(config.no_of_port):
        df['port_load%i' % i] = LOAD[i]
    for i in range(config.no_of_port):
        for j in range(config.no_of_buffer):
            df['load_dst{}_{}'.format(i, j)] = C_dst_SET[(i, j)]

    #arrival patterns
    df['inter_arr_sys'] = df['timestamp (sec)'].diff()
    if config.no_of_buffer > 1:
        for i in range(config.no_of_buffer):
            t = df[df['priority'] ==
                    i]['timestamp (sec)'].diff().rename(
                        'inter_arr{}'.format(i)).to_frame()
            df = df.join(t)

    #save
    filename = file.split('/')[-1]
    drop_cols = ['timestamp (sec)'] + target_col
    if config.no_of_buffer == 1: drop_cols += ['priority']
    fet_cols = list(df.columns.drop(drop_cols))
    # my_fet['fet_cols'] = fet_cols
    processed_df = df[fet_cols + target_col].fillna(method='ffill').dropna()
    # if save:
    #     processed_df.to_csv('{}/{}'.format(dst_folder, filename), index=False)

    key = filename.split(".csv")[0]
    ins = build_timeseries(processed_df.values, target_col_index=[-1])
    x, y = ins.timeseries(config.TIME_STEPS)
    "randomly selected part of the them. to represent the config"
    loc = np.random.choice(
        np.arange(len(y)),
        np.max([np.min([14000, len(y)]),
                int(len(y) * 0.15)]),
        replace=False)

    def write_hdf(file, x, y):
        with h5py.File(file, 'w') as hdf:
            hdf['x'] = x
            hdf['y'] = y

    x = x[loc]
    y = y[loc]
    if mode == 'train':
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=config.test_size, shuffle=True)
        write_hdf(
            '{}/_hdf/train/{}.h5'.format(base_dir, key), 
            x_train, y_train)
        write_hdf(
            '{}/_hdf/valid/{}.h5'.format(base_dir, key), 
            x_test, y_test)
    else:
        write_hdf(
            '{}/_hdf/test/{}.h5'.format(base_dir, key), 
            x, y)


# @numba.jit
def process_trace_return(file, base_dir, mode):
    config = BaseConfig()
    target_col = ['time_in_sys']

    df = pd.read_csv(file).fillna(config.sp_wgt)

    #traffic load features
    ins = feaTure(df, config.no_of_port, config.no_of_buffer,
                    config.window, config.ser_rate)
    C_dst_SET, LOAD = ins.getCount()
    for i in range(config.no_of_port):
        df['port_load%i' % i] = LOAD[i]
    for i in range(config.no_of_port):
        for j in range(config.no_of_buffer):
            df['load_dst{}_{}'.format(i, j)] = C_dst_SET[(i, j)]

    #arrival patterns
    df['inter_arr_sys'] = df['timestamp (sec)'].diff()
    if config.no_of_buffer > 1:
        for i in range(config.no_of_buffer):
            t = df[df['priority'] ==
                    i]['timestamp (sec)'].diff().rename(
                        'inter_arr{}'.format(i)).to_frame()
            df = df.join(t)

    #save
    filename = file.split('/')[-1]
    drop_cols = ['timestamp (sec)'] + target_col
    if config.no_of_buffer == 1: drop_cols += ['priority']
    fet_cols = list(df.columns.drop(drop_cols))
    # my_fet['fet_cols'] = fet_cols
    processed_df = df[fet_cols + target_col].fillna(method='ffill').dropna()
    # if save:
    #     processed_df.to_csv('{}/{}'.format(dst_folder, filename), index=False)

    key = filename.split(".csv")[0]
    ins = build_timeseries(processed_df.values, target_col_index=[-1])
    x, y = ins.timeseries(config.TIME_STEPS)
    "randomly selected part of the them. to represent the config"
    loc = np.random.choice(
        np.arange(len(y)),
        np.max([np.min([14000, len(y)]),
                int(len(y) * 0.15)]),
        replace=False)

    def write_hdf(file, x, y):
        with h5py.File(file, 'w') as hdf:
            hdf['x'] = x
            hdf['y'] = y
    
    x_org = x
    y_org = y

    x = x[loc]
    y = y[loc]
    if mode == 'train':
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=config.test_size, shuffle=True)
        write_hdf(
            '{}/_hdf/train/{}.h5'.format(base_dir, key), 
            x_train, y_train)
        write_hdf(
            '{}/_hdf/valid/{}.h5'.format(base_dir, key), 
            x_test, y_test)
    else:
        write_hdf(
            '{}/_hdf/test/{}.h5'.format(base_dir, key), 
            x, y)
    
    return x_org, y_org
    


class PreporcessDataset:

    def __init__(self, data_dir, mode = "train"):
        self.data_dir = data_dir
        self.config = BaseConfig()
        self.mode = mode
        
        file_dir = '{}/{}/_traces/_{}'.format(
            self.data_dir,
            self.config.modelname, "train" if mode == "valid" else mode)
        
        FILES = []
        for dirpath, dirnames, filenames in os.walk(file_dir):
            for file in filenames:
                if (os.path.splitext(file)[1]
                        == '.csv') and 'checkpoint' not in file:
                    FILES.append(os.path.join(dirpath, file))

        self.data_files = FILES

        dst_folder_name = "train_preprocessed"
        self.dst_folder = '{}/{}/_traces/{}'.format(
                self.data_dir,
                self.config.modelname, dst_folder_name)

        self.target = ['time_in_sys']
        self.fet_cols = None

        print("Mode: {}, data file num: {}".format(self.mode, len(self.data_files)))
    
    def get_csv_file_example(self, idx):
        file = self.data_files[idx]
        df = pd.read_csv(file).fillna(self.config.sp_wgt)
        return df
    
    def get_h5_file_example(self, idx):
        csv_file = self.data_files[idx]
        filename = csv_file.split('/')[-1]
        key = filename.split(".csv")[0]
        h5_file = '{}/_hdf/train/{}.h5'.format(self.data_dir, key)
        return self.load_hdf(h5_file)


    def multi_process(self, func, FILES, args=None):
        it = 0
        while True:
            files = FILES[it * self.config.no_process:(it + 1) *
                            self.config.no_process]
            if len(files) > 0:
                threads = []
                for file in files:
                    ARGS = list(args)
                    t = Process(target=func, args=tuple([file] + ARGS))
                    threads.append(t)
                    t.start()
                for thr in threads:
                    thr.join()
                it += 1
            else:
                break
    
    def build_folder(self, reset = False):
        new_folders = [
                '_hdf', '_hdf/train', '_hdf/valid', '_hdf/test', '_scaler',
                '_processed', '_processed/train', '_processed/valid', '_processed/test',
                '_error'
        ]
        for name in new_folders:
            folder = '{}/{}'.format(self.data_dir, name)
            if reset:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)
            else:
                if not os.path.exists(folder):
                    os.makedirs(folder)
    
    def preprocess_example(self, idx):
        file = self.data_files[idx]
        x,y = process_trace_return(file, self.data_dir, "")
        return x,y
    
    def normalize_example(self, x, y):
        x = (x - self.x_MIN.values) / (self.x_MAX.values -
                                           self.x_MIN.values)
        y = (y - self.y_MIN.values) / (self.y_MAX.values -
                                        self.y_MIN.values)
        return x, y 


    def data_preprocess(self):
        print("Start data preprocess...")
        # print(self.data_files[:10])

        self.multi_process(process_trace, self.data_files, args=(self.data_dir, self.mode))
        
        print("Data preprocess done...")
    
    def load_hdf(self, file):
        with h5py.File(file, 'r') as hdf:
            x = hdf['x'][:]
            y = hdf['y'][:]
        return x, y
    
    def get_fet_cols(self):
        config = self.config
        target_col = self.target
        
        file = self.data_files[0]

        df = pd.read_csv(file).fillna(config.sp_wgt)
        #traffic load features
        ins = feaTure(df, config.no_of_port, config.no_of_buffer,
                        config.window, config.ser_rate)
        C_dst_SET, LOAD = ins.getCount()
        for i in range(config.no_of_port):
            df['port_load%i' % i] = LOAD[i]
        for i in range(config.no_of_port):
            for j in range(config.no_of_buffer):
                df['load_dst{}_{}'.format(i, j)] = C_dst_SET[(i, j)]

        #arrival patterns
        df['inter_arr_sys'] = df['timestamp (sec)'].diff()
        if config.no_of_buffer > 1:
            for i in range(config.no_of_buffer):
                t = df[df['priority'] ==
                        i]['timestamp (sec)'].diff().rename(
                            'inter_arr{}'.format(i)).to_frame()
                df = df.join(t)

        #save
        drop_cols = ['timestamp (sec)'] + target_col
        if config.no_of_buffer == 1: drop_cols += ['priority']
        self.fet_cols = list(df.columns.drop(drop_cols))
    
    def sample_validation_data(self):
        x,y  = self.load_merged_data(mode = "valid")
        loc = np.random.choice(np.arange(len(y)),
                                       int(len(y) * self.config.sub_rt),
                                       replace=False)
        x_valid = x[loc]
        y_valid = y[loc]
        with h5py.File(
                '{}/_processed/{}/sampled_{}.h5'.format(self.data_dir, "valid", "valid"),
                'w') as hdf:
                hdf['x'] = x_valid
                hdf['y'] = y_valid

    def cal_min_max(self):
        """cal. data_range from the training dataset."""
        src = '{}/_hdf/train/*.h5'.format(self.data_dir)

        for i, file in enumerate(glob.glob(src)):
            x, y = self.load_hdf(file)
            # fet_cols = list(range(x.shape[-1]))
            # target = [-1]
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
        ins.save('{}/_scaler'.format(self.data_dir))
        cScaler(self.y_MIN, self.y_MAX,
                'y').save('{}/_scaler'.format(self.data_dir))
    
    def load_scaler(self):
        folder = '{}/_scaler'.format(self.data_dir)
        self.x_MIN, self.x_MAX, self.y_MIN, self.y_MAX, self.fet_cols, self.target = load_scaler(
            folder)
    
    def load_merged_data(self, mode=None, save_name=None):
        if mode is None:
            mode = self.mode
        if save_name is None:
            save_name  = mode
        
        file = "{}/_processed/{}/merged_{}.h5".format(self.data_dir, mode, save_name)
        with h5py.File(file, "r") as hdf:
            x = hdf['x'][:]
            y = hdf['y'][:]
        return x, y

    def merge_and_normalize(self, mode=None, save_name=None):
        if save_name is None:
            save_name  = self.mode
        if mode is None:
            mode = self.mode
        src = '{}/_hdf/{}/*.h5'.format(self.data_dir, mode)
        all_x = []
        all_y = []

        for file in glob.glob(src):
            with h5py.File(file, "r") as hdf:
                one_x = hdf['x'][:]
                one_y = hdf['y'][:]
                all_x.append(one_x)
                all_y.append(one_y)
                
        x = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)

        x = (x - self.x_MIN.values) / (self.x_MAX.values -
                                           self.x_MIN.values)
        y = (y - self.y_MIN.values) / (self.y_MAX.values -
                                        self.y_MIN.values)

        with h5py.File(
                '{}/_processed/{}/merged_{}.h5'.format(self.data_dir, mode, save_name),
                'w') as hdf:
                hdf['x'] = x
                hdf['y'] = y
    
    def h5py_to_torch(self, mode=None, data_save_name=None):
        if mode is None:
            mode = self.mode
        if data_save_name is None:
            data_save_name = self.mode
        
        file_path = '{}/_processed/{}/merged_{}.h5'.format(
            self.data_dir,
            mode,
            data_save_name)
        
        data_save_path  = '{}/_processed/{}/merged_{}_data.pt'.format(
            self.data_dir,
            self.mode,
            data_save_name)
        label_save_path  = '{}/_processed/{}/merged_{}_label.pt'.format(
            self.data_dir,
            self.mode,
            data_save_name)
        
        with h5py.File(file_path, "r") as hdf:
            data, label = hdf['x'][:], hdf['y'][:]
        
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        
        torch.save(data, data_save_path)
        torch.save(label, label_save_path)

    def sampled_valid_h5py_to_torch(self):
        
        file_path = '{}/_processed/{}/sampled_{}.h5'.format(
            self.data_dir,
            "valid",
            "valid")
        
        data_save_path  = '{}/_processed/{}/sampled_{}_data.pt'.format(
            self.data_dir,
            "valid",
            "valid")
        label_save_path  = '{}/_processed/{}/sampled_{}_label.pt'.format(
            self.data_dir,
            "valid",
            "valid")
        
        with h5py.File(file_path, "r") as hdf:
            data, label = hdf['x'][:], hdf['y'][:]
        
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        
        torch.save(data, data_save_path)
        torch.save(label, label_save_path)
        

class H5Dataset(Dataset):
    def __init__(self, data_dir, gap = 1, mode="train", data_save_name = None):
        self.data_dir = data_dir
        self.config = BaseConfig()

        pre_name = "sampled" if "sampled" in mode else "merged"
        mode = "valid" if mode == "sampled_valid" else mode

        self.mode = mode
        if data_save_name is None:
            self.data_save_name = self.mode
        else:
            self.data_save_name = data_save_name
        
        self.data_file_path = '{}/_processed/{}/{}_{}_data.pt'.format(
            self.data_dir,
            self.mode,
            pre_name,
            self.data_save_name)
        
        self.label_file_path = '{}/_processed/{}/{}_{}_label.pt'.format(
            self.data_dir,
            self.mode,
            pre_name,
            self.data_save_name)
        
        if gap > 1:
            self.data = torch.load(self.data_file_path)[::gap]
            self.label = torch.load(self.label_file_path)[::gap]
        else:
            self.data = torch.load(self.data_file_path)
            self.label = torch.load(self.label_file_path)
            
        assert self.data.size()[0] == self.label.size()[0], "ERROR: self.data and self.label not of equal length"
        
        self.data = self.data #- 0.5



    def __len__(self):
        return self.data.size()[0] 
    
    def __getitem__(self, idx):
        return self.data[idx].cuda(), self.label[idx].cuda()
    



if __name__ == '__main__':
    print("Start data preprocessing ...")
    base_dir = "./DeepQueueNet-synthetic-data/data"

    dataset = PreporcessDataset(base_dir, mode="train")
    dataset.build_folder()
    dataset.data_preprocess()
    dataset.get_fet_cols()
    dataset.sample_validation_data()
    dataset.sampled_valid_h5py_to_torch()
    print("Preprocessing done ...")