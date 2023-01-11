import os, sys
import pandas as pd 
import numpy as np 
import time
import copy

import multiprocessing as mp
from multiprocessing import Process, Manager, connection



import torch

from tf_impl.minmaxscaler import load_scaler
from tf_impl.trafficindicator import feaTure 
from tf_impl.error_corr import  get_error_dis, error_correction


from config import BaseConfig, ModelConfig

from model import DeviceModel, DeviceModelBiLSTM

import warnings
warnings.filterwarnings("ignore")

class fattree:

    def __init__(self, 
                file_idx,
                traffic_pattern = "poisson",
                base_dir = "./DeepQueueNet-synthetic-data/data",
                model_identifier = "default"
                ):

        self.config = BaseConfig()
        self.model_config = ModelConfig()

        self.base_dir = base_dir

        # traffic_pattern = "poisson" #"onoff" #"map"
        # file_idx = 0 # 0 to 4
        self.input_file = '{}/fattree16/{}/rsim{}'.format(self.base_dir, traffic_pattern, file_idx+1)
        flow = pd.read_csv('{}.csv'.format(self.input_file)).reset_index(drop=True)
        flow['src_hub']=flow['src_pc'] // 2 + 12
        flow['src_port']=flow['src_pc'] % 2

        self.flow = self.init_flow(flow)
        print("flow loading and initialization done ...")
        print("flow data shape:", self.flow.shape)

        if self.model_config.error_correction:
            self.ERR, self.BINS = get_error_dis(self.config, self.model_config)

        self.port_connection_matrix = \
                   [[-1,-1,-1,-1,0,-1,1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,0,-1,1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,0,-1,1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,0,-1,1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1],
                    [2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1,-1,-1,-1,-1],
                    [2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1,-1,-1],
                    [-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1,-1,-1],
                    [2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1],
                    [-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,-1,-1],
                    [2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1],
                    [-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1],
                    [-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,3,-1,-1,-1,-1,-1,-1,-1,-1]]
        
        self.port_connection = dict()

        self.model_identifier = model_identifier
        self.saved_model = "saved/{}/saved_model/best_model.pt".format(self.model_identifier)

        self.model = None
        self.model_path = self.saved_model

    def init_flow(self, flow):
        flow['dep_time'] = flow['etime']
        flow['etime'] = flow['timestamp (sec)']
        flow['cur_hub'] = flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[0]))
        flow['cur_port'] = flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
        flow['status'] = 0   #times the packet has been forwarded in the sys.
        return flow
    
    def parse_topo(self) -> None:
        for from_node, to_list in enumerate(self.port_connection_matrix):
            for to_node, to_port in enumerate(to_list):
                if from_node not in self.port_connection:
                    self.port_connection[from_node] = dict()
                if to_port >= 0:    
                    self.port_connection[from_node][to_port] = to_node

    def model_input(self, df, layer):
        dt = df.copy()
        dt['timestamp (sec)'] = dt['etime']
        dt['src'] = dt[['src_port', 'status', 'path']].apply(self.get_src, axis=1)
        dt['dst'] = dt['cur_port']
        self.add_egress_feature(dt)
        self.add_ingress_fetature(dt)
        dt=dt.fillna(method='ffill').fillna(method='bfill').fillna(0.)
        return dt[self.fet_cols].copy()

    def get_src(self, x): 
        src_port, status, path = x[0], x[1], x[2]
        if status == 0: 
            return src_port 
        else: 
            last_hub = int(path.split('-')[status-1].split('_')[0])
            last_layer = 'core' if last_hub//4 == 0 else 'agg' if last_hub//12 == 0 else 'edge'
            cur_hub = int(path.split('-')[status].split('_')[0])
            cur_layer = 'core' if cur_hub//4 == 0 else 'agg' if cur_hub//12 == 0 else 'edge'
            if last_layer == 'core':
                #downward forwarding 
                return 2+last_hub%2
            elif last_layer == 'agg':
                if cur_layer == 'core':
                    #upward forwarding
                    return (last_hub-4)//2
                else:
                    #downward forwarding 
                    return 2+last_hub%2
            else:
                #upward forwarding 
                return last_hub%2

    def load_scaler(self):
        self.x_MIN, self.x_MAX, self.y_MIN, self.y_MAX, self.fet_cols, _ = load_scaler("{}/_scaler".format(self.base_dir))
    
    def load_model(self, return_model = False):
        seq_len = self.config.TIME_STEPS
        input_dim = 12
        embed_dim = 200
        hidden_dim = 100
        output_dim = 1
        
        self.model = DeviceModel(seq_len, input_dim, embed_dim, hidden_dim, output_dim)
        # self.model = DeviceModelBiLSTM(seq_len, input_dim, embed_dim, hidden_dim, output_dim)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        if return_model:
            return copy.deepcopy(self.model)



    def add_ingress_fetature(self, df):
        df['inter_arr_sys'] = df['timestamp (sec)'].diff()

    def add_egress_feature(self, df):
        ins = feaTure(df, 
                    self.config.no_of_port,  
                    self.config.no_of_buffer, 
                    self.config.window, 
                    self.config.ser_rate)
        Count_dst_SET, LOAD = ins.getCount()
        for i in range(self.config.no_of_port): 
            df['port_load%i' %i] = LOAD[i] 
        for i in range(self.config.no_of_port):
            for j in range(self.config.no_of_buffer):
                df['load_dst%i_%i' %(i,j)] = np.array(Count_dst_SET[(i,j)])
    
    def timeseries(self, sample):
        #total number of time-series samples would be x.shape[0]-timesteps+1
        dim0 = sample.shape[0]-self.config.TIME_STEPS+1
        dim1 = sample.shape[1]
        x = np.zeros((dim0, self.config.TIME_STEPS, dim1))
        for i in range(dim0): 
            x[i] = sample[i:self.config.TIME_STEPS+i]
        return x
    
    def flows_to_port(self):
        self.IN = dict()
        for i in range(20): 
            self.IN['DataFrame{}'.format(i)] = self.flow[self.flow.src_hub==i].copy().sort_values('etime').reset_index(drop=True) 
    

    def predict(self, data, model, device, batch_size = 2048):
        out_list = []

        data = torch.from_numpy(data).float() #- 0.5 
        # minus 0.5 because in H5Dataset, data was decreased by 0.5 for faster model convergence
        
        with torch.no_grad():
            data_len = len(data)
            for start_idx in range(0, data_len, batch_size):
                end_idx = start_idx + batch_size if start_idx + batch_size <= data_len else data_len
                batch = data[start_idx:end_idx].to(device)
                out = model(batch)
                out_list.append(out.cpu().flatten())
        
        return_val = torch.cat(out_list)
        
        assert len(return_val) == len(data)

        return return_val.numpy() 

    def infer(self, flow,  layer, model, device):
        """This module is to predict the time a packet spends in a device.
           Input:     
             - flow:    flow to a device
             - layer:   which layer the device is located
             - model:   model used in this process to conduct delay inference
             - device:  the device used in this process (a pytorch device object)
        """
        #flow -> X
        input_dt = self.model_input(flow, layer)  
        #X->normalize(X)
        input_dt = (input_dt-self.x_MIN)/(self.x_MAX-self.x_MIN) 
        #X-> batches of timeseries
        x_input = self.timeseries(input_dt.values)
        #infer.
        y_pred = self.predict(x_input, model, device)#.flatten()

        # print(self.y_MIN)
        y_pred = self.y_MIN[-1] + y_pred*(self.y_MAX[-1] - self.y_MIN[-1])
        flow = flow.iloc[self.config.TIME_STEPS-1:]
        #upd. departure time
        if self.model_config.error_correction:
            flow['delay'] = y_pred
            flow = error_correction(flow, self.config, self.ERR, self.BINS)
            y_pred = [max(x, float(y)/self.config.ser_rate) for x, y in zip(flow.delay.values, flow['pkt len (byte)'].values)]
            flow['etime'] = flow['etime'] + y_pred   
            flow.drop(['delay', 'bins'], axis=1, inplace=True)
        else:
            y_pred = [max(x, float(y)/self.config.ser_rate) for x, y in zip(y_pred, flow['pkt len (byte)'].values)]
            flow['etime'] = flow['etime'] + y_pred 
        return flow

    def infer_multi_nodes(self, flow_dict,  layer, model, device):
        """This module is to predict the time a packet spends in a device.
           Input:     
             - flow_dict:    flow to devices with device index as the key and flow to the device as value in the dict 
             - layer:   which layer the device is located
             - model:   model used in this process to conduct delay inference
             - device:  the device used in this process (a pytorch device object)
        """
        flow_len_dict = dict()
        all_input_list = []

        # print(flow_dict)

        for node_id, flow in flow_dict.items():
            #flow -> X
            input_dt = self.model_input(flow, layer)  
            #X->normalize(X)
            input_dt = (input_dt-self.x_MIN)/(self.x_MAX-self.x_MIN) 
            #X-> batches of timeseries
            x_input = self.timeseries(input_dt.values)

            flow_len_dict[node_id] = len(x_input)
            all_input_list.append(x_input)

        x_input_all = np.concatenate(all_input_list)
        #infer.
        y_pred_all = self.predict(x_input_all, model, device)#.flatten()

        # print(self.y_MIN)
        y_pred_all = self.y_MIN[-1] + y_pred_all*(self.y_MAX[-1] - self.y_MIN[-1])
        
        return_flow_dict = dict()
        start_pos = 0
        end_pos = 0
        for node_id, flow_len in flow_len_dict.items():
            flow = flow_dict[node_id]

            end_pos += flow_len

            y_pred = y_pred_all[start_pos:end_pos]

            flow = flow.iloc[self.config.TIME_STEPS-1:]

            start_pos = end_pos

            #upd. departure time
            if self.model_config.error_correction:
                flow['delay'] = y_pred
                flow = error_correction(flow, self.config, self.ERR, self.BINS)
                y_pred = [max(x, float(y)/self.config.ser_rate) for x, y in zip(flow.delay.values, flow['pkt len (byte)'].values)]
                flow['etime'] = flow['etime'] + y_pred   
                flow.drop(['delay', 'bins'], axis=1, inplace=True)
            else:
                y_pred = [max(x, float(y)/self.config.ser_rate) for x, y in zip(y_pred, flow['pkt len (byte)'].values)]
                flow['etime'] = flow['etime'] + y_pred 
            
            return_flow_dict[node_id] = flow

        return return_flow_dict 

    def link_info_upd(self, link): 
        #forward to another hub/join another Engress port
        link['status'] += 1
        link['cur_hub'] = link[['status','path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[0]), axis=1)
        link['cur_port'] = link[['status','path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[1]), axis=1)
        return link

    def trace_upd(self, layer, i, my_result, my_link, model, device):
        """This module is to update my_result/my_link.
           Input:
             - layer:     which layer the device is located
             - i:         which position the device is located
             - my_result: shared result among GPUs  
             - my_link:   shared link traffic among GPUs  
             - model:     which model used for the prediction in this process
             - device:    the pytorch device for this process
        """
        #collect traffic to a device
        trace_in = self.IN['DataFrame{}'.format(i)].copy()  
        for c in range(20):
            link_data = my_link[(c,i)]  
            if link_data.shape[0] > 0:
                trace_in = trace_in.append(link_data, ignore_index=True)
        trace_in = trace_in.sort_values('etime').reset_index(drop=True)
        #infer departure time  
        trace_out = self.infer(trace_in,
                             layer, model, device)
        #upd. my_link
        for connected_node, connected_port in enumerate(self.port_connection_matrix[i]): 
            if connected_port > -1: 
                eport_flow=trace_out[trace_out.cur_port == connected_port]
                if eport_flow.shape[0] > 0: 
                    my_link[(i,connected_node)] = self.link_info_upd(eport_flow)
        #upd. my_result
        if layer=='edge':
            out_port0 = my_result[(i-12)*2].append(trace_out[trace_out.cur_port == 0][self.used_cols], ignore_index=True)
            out_port1 = my_result[(i-12)*2+1].append(trace_out[trace_out.cur_port == 1][self.used_cols], ignore_index=True)
            my_result[(i-12)*2] = out_port0
            my_result[(i-12)*2+1] = out_port1

    def trace_upd_multi_nodes(self, layer, node_list, my_result, my_link, model, device):
        """This module is to update my_result/my_link.
           Input:
             - layer:     which layer the device is located
             - node_list: position list for the device nodes
             - my_result: shared result among GPUs  
             - my_link:   shared link traffic among GPUs  
             - model:     which model used for the prediction in this process
             - device:    the pytorch device for this process
        """
        #collect traffic to a device
        def collect_trace_in(i : int):
            trace_in = self.IN['DataFrame{}'.format(i)].copy()  
            for c in range(20):
                link_data = my_link[(c,i)]  
                if link_data.shape[0] > 0:
                    trace_in = trace_in.append(link_data, ignore_index=True)
            trace_in = trace_in.sort_values('etime').reset_index(drop=True)
            return trace_in

        all_trace_dict = dict()
        for node_id in node_list:
            all_trace_dict[node_id] = collect_trace_in(node_id)
        

        #infer departure time  
        trace_out_dict = self.infer_multi_nodes(all_trace_dict,
                             layer, model, device)
        
        # update ingress traffic over the inwards links for one node
        def update_one_node(i : int, trace_out):
            #upd. my_link
            for connected_node, connected_port in enumerate(self.port_connection_matrix[i]): 
                if connected_port > -1: 
                    eport_flow=trace_out[trace_out.cur_port == connected_port]
                    if eport_flow.shape[0] > 0: 
                        my_link[(i,connected_node)] = self.link_info_upd(eport_flow)
            #upd. my_result
            if layer=='edge':
                out_port0 = my_result[(i-12)*2].append(trace_out[trace_out.cur_port == 0][self.used_cols], ignore_index=True)
                out_port1 = my_result[(i-12)*2+1].append(trace_out[trace_out.cur_port == 1][self.used_cols], ignore_index=True)
                my_result[(i-12)*2] = out_port0
                my_result[(i-12)*2+1] = out_port1
        
        for node_id in node_list:
            update_one_node(node_id, trace_out_dict[node_id])
    



    #deploy the task on multi-GPUs. Each GPU excutes the inference on a sub-graph.
    def pod_infer(self, gpu_number, device_idx, my_result, my_link):
        '''
        This function will be used to start-up the process for each GPU device.

        - gpu_number:   number for GPUs used
        - device_idx:   the index for the GPU used in this process
        - my_result:    shared result among GPUs
        - my_link:      shared link traffic among GPUs 
        '''
        # The CUDA_VISIBLE_DEVICE env var should not be modified in pytorch implementation.
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(device_idx)
        
        if self.model is None:
            model = self.load_model(return_model = True)
        else:
            model = copy.deepcopy(self.model)
        
        if gpu_number == 1:
            core_bg, core_ed= 0,4
            agg_bg, agg_ed= 4,12
            edge_bg, edge_ed=12,20 
            pc_bg, pc_ed=0,16 
        elif gpu_number == 2:
            core_bg, core_ed=((0,2), (2,4))[device_idx]
            agg_bg, agg_ed=((4,8), (8,12))[device_idx]
            edge_bg, edge_ed=((12,16), (16,20))[device_idx]
            pc_bg, pc_ed=((0,8), (8,16))[device_idx]
        elif gpu_number == 4:
            core_bg, core_ed=((0,1), (1,2), (2,3), (3,4))[device_idx]
            agg_bg, agg_ed=((4,6), (6,8), (8,10), (10,12))[device_idx]
            edge_bg, edge_ed=((12,14), (14,16), (16,18), (18,20))[device_idx]
            pc_bg, pc_ed=((0,4), (4,8), (8,12), (12,16))[device_idx]
        else:
            raise Exception("GPU number not correct as %d" % (gpu_number))
        
        device = torch.device(device_idx)
        model = model.to(device)
        #IRSA
        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols)
        for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, my_result, my_link, model, device)
        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols) #clear the unstable results
        for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, my_result, my_link, model, device)
        for i in range(core_bg, core_ed): self.trace_upd('core', i, my_result, my_link, model, device) 
        for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, my_result, my_link, model, device)
        for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, my_result, my_link, model, device)
    

    def run(self, gpu_number=1):
        '''
        Run the simulation with GPU number specified.
        '''

        print("GPU number:", gpu_number)
    
        self.used_cols=['index', 
                        'timestamp (sec)', 
                        'pkt len (byte)',  
                        'priority',  
                        'src_hub','src_port',
                        'cur_hub','cur_port',
                        'path', 
                        'dep_time', 
                        'etime']

        self.load_scaler()
        self.flows_to_port()

        LINKS = {(i,j): pd.DataFrame() for i in range(20) for j in range(20)} 
       
        with Manager() as MG:
            my_result = MG.dict() 
            my_link = MG.dict(LINKS)
                            
            #parallel reasoning                
            t0 = time.time()         
            threads = []
            for i in range(gpu_number): 
                t = Process(target=self.pod_infer, args=(gpu_number, i, my_result, my_link))
                threads.append(t)
                t.start()
            for thr in threads:  thr.join()
                
            
            print("time used (total)", "%f min." %((time.time() - t0) / 60))
            result = pd.DataFrame()
            for key in my_result.keys(): 
                result = pd.concat([result, my_result[key]], ignore_index=True)
            
            result.reset_index(drop=True).to_csv('{}_pred.csv'.format(self.input_file), index=False)
    
    
    #deploy the task on multi-GPUs. Each GPU excutes the inference on a sub-graph.
    def pod_infer_sync(self, gpu_number, device_idx, my_result, my_link, my_progress):
        '''
        This function will be used to start-up the process for each GPU device.

        - gpu_number:   number for GPUs used
        - device_idx:   the index for the GPU used in this process
        - my_result:    shared result among GPUs
        - my_link:      shared link traffic among GPUs 
        - my_progree:   shared progress dict which will record the progress as int for each GPU-process,
                        this variable is used to synchronize the progress of all GPUs
        '''
        # The CUDA_VISIBLE_DEVICE env var should not be modified in pytorch implementation.
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(device_idx)

        print("Pod infer sync idx: ", device_idx)
        
        if self.model is None:
            model = self.load_model(return_model = True)
        else:
            model = copy.deepcopy(self.model)
        
        if gpu_number == 1:
            core_bg, core_ed= 0,4
            agg_bg, agg_ed= 4,12
            edge_bg, edge_ed=12,20 
            pc_bg, pc_ed=0,16 
        elif gpu_number == 2:
            core_bg, core_ed=((0,2), (2,4))[device_idx]
            agg_bg, agg_ed=((4,8), (8,12))[device_idx]
            edge_bg, edge_ed=((12,16), (16,20))[device_idx]
            pc_bg, pc_ed=((0,8), (8,16))[device_idx]
        elif gpu_number == 4:
            core_bg, core_ed=((0,1), (1,2), (2,3), (3,4))[device_idx]
            agg_bg, agg_ed=((4,6), (6,8), (8,10), (10,12))[device_idx]
            edge_bg, edge_ed=((12,14), (14,16), (16,18), (18,20))[device_idx]
            pc_bg, pc_ed=((0,4), (4,8), (8,12), (12,16))[device_idx]
        else:
            raise Exception("GPU number not correct as %d" % (gpu_number))
        
        def sync(gpu_number, my_progress,  idx):
            cur_progress = my_progress[idx]

            while True:
                wait = False
                for i in range(gpu_number):
                    if my_progress[i] < cur_progress:
                        wait = True
                        break
                
                if not wait:
                    break
                else:
                    time.sleep(0.01)
            
            # print("process", idx, "progress", my_progress[idx])
            

        device = torch.device(device_idx)
        model = model.to(device)
        #IRSA
        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols)
        # for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, my_result, my_link, model, device)
        self.trace_upd_multi_nodes('edge', list(range(edge_bg, edge_ed)), my_result, my_link, model, device)
        
        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)
        
        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols) #clear the unstable results
        # for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, my_result, my_link, model, device)
        self.trace_upd_multi_nodes('agg', list(range(agg_bg, agg_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        # for i in range(core_bg, core_ed): self.trace_upd('core', i, my_result, my_link, model, device) 
        self.trace_upd_multi_nodes('core', list(range(core_bg, core_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        # for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, my_result, my_link, model, device)
        self.trace_upd_multi_nodes('agg', list(range(agg_bg, agg_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        # for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, my_result, my_link, model, device)
        self.trace_upd_multi_nodes('edge', list(range(edge_bg, edge_ed)), my_result, my_link, model, device)



    def run_parallel(self, gpu_number=4):
        '''
        Run the simulation with GPU number specified in **parallel**.
        '''
        print("Run parallel")

        print("GPU number:", gpu_number)
    
        self.used_cols=['index', 
                        'timestamp (sec)', 
                        'pkt len (byte)',  
                        'priority',  
                        'src_hub','src_port',
                        'cur_hub','cur_port',
                        'path', 
                        'dep_time', 
                        'etime']

        self.load_scaler()
        self.flows_to_port()

        LINKS = {(i,j): pd.DataFrame() for i in range(20) for j in range(20)} 
       
        with Manager() as MG:
            my_result = MG.dict() 
            my_link = MG.dict(LINKS)

            my_progress = MG.list([0 for _ in range(gpu_number)])

                            
            #parallel reasoning                
            t0 = time.time()         
            threads = []
            for i in range(gpu_number): 
                t = Process(target=self.pod_infer_sync, args=(gpu_number, i, my_result, my_link, my_progress))
                threads.append(t)
                t.start()
            for thr in threads:  thr.join()
                
            
            print("time used (total)", "%f min." %((time.time() - t0) / 60))
            result = pd.DataFrame()
            for key in my_result.keys(): 
                result = pd.concat([result, my_result[key]], ignore_index=True)
            
            result.reset_index(drop=True).to_csv('{}_pred.csv'.format(self.input_file), index=False)


    



if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    for traffic_pattern in [
        "poisson" ,
        "onoff" ,
        "map"
        ]:
        for idx in [
            0,
            1,
            2,
            3,
            4
        ]:
            print("Traffic: {}, Run idx: {}".format(traffic_pattern, idx))
            ft = fattree(idx, traffic_pattern=traffic_pattern)
            ft.run_parallel(gpu_number=1)
            print()
    



    

    
