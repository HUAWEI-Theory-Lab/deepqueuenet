# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache-2.0 License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache-2.0 License for more details.

import tensorflow as tf 
import os
import pandas as pd 
import numpy as np 
import time
from multiprocessing import Process, Manager
from .topo import  Init_TOPO
from code_deepQueueNet.tools.error_corr import  get_error_dis, error_correction
import warnings
warnings.filterwarnings("ignore")




"""
This script is to implement the inference on fattree16.   
INDEXs - PCs: from left to right are labeled from 0 to 15.
       - Switchs:
         1. core-layer: from left to right, are labeled from 0 to 3
         2. aggregation-layer: from left to right, are labeled from 4 to 11
         3. edge-layer: from left to right, are labeled from 12 to 19
       - PORT NUMBERs of a switch: from bottom to top, left to right, are labeled 0,1,2 and 3 respectively.
FUNCTIONS:
     1. model_input       traffic -> feature extraction
     2. infer             deepQueueNet
     3. link_info_upd       
     4. trace_upd         device-level inference   
     5. run               topology-level inference in parallel  
"""




class INFER(Init_TOPO):
    def __init__(self, filename, config, model_config):
        self.filename=filename
        flow=pd.read_csv('{}.csv'.format(filename)).reset_index(drop=True)
        flow['src_hub']=flow['src_pc']//2+12
        flow['src_port']=flow['src_pc']%2
        super(INFER, self).__init__(flow, config, model_config)
        if model_config.error_correction:
            self.ERR, self.BINS=get_error_dis(config, model_config)
         
         
        
        
 

    def model_input(self, df, layer):
        dt=df.copy()
        dt['timestamp (sec)']=dt['etime']
        dt['src']=dt[['src_port', 'status', 'path']].apply(self.get_src, axis=1)
        dt['dst']=dt['cur_port']
        self.addEgressFet(dt)
        self.addIngressFet(dt)
        dt=dt.fillna(method='ffill').fillna(method='bfill').fillna(0.)
        return dt[self.fet_cols].copy()
    
    
    
    
    
    
    
 

    def infer(self, flow,  layer, sess, outputs, X):
        """This module is to predict the time a packet spends in a device.
           Input:     
             - flow:    flow to a device
             - layer:   which layer the device is located
             - sess:    trained deepQueueNet for inference
             - outputs: delay
             - X:       features
        """
        #flow -> X
        input_dt=self.model_input(flow, layer)  
        #X->normalize(X)
        input_dt=(input_dt-self.x_MIN)/(self.x_MAX-self.x_MIN)
        #X-> batches of timeseries
        x_input=self.timeseries(input_dt.values)
        #infer.
        y_pred=sess.run(outputs, feed_dict={X:  x_input}).flatten() 
        y_pred=self.y_MIN['time_in_sys']+y_pred*(self.y_MAX['time_in_sys']-self.y_MIN['time_in_sys'])
        flow=flow.iloc[self.config.TIME_STEPS-1:]
        #upd. departure time
        if self.model_config.error_correction:
            flow['delay']=y_pred
            flow=error_correction(flow, self.config, self.ERR, self.BINS)
            y_pred=[max(x, float(y)/self.config.ser_rate) for x, y in zip(flow.delay.values, flow['pkt len (byte)'].values)]
            flow['etime']=flow['etime']+y_pred   
            flow.drop(['delay', 'bins'], axis=1, inplace=True)
        else:
            y_pred=[max(x, float(y)/self.config.ser_rate) for x, y in zip(y_pred, flow['pkt len (byte)'].values)]
            flow['etime']=flow['etime']+y_pred 
        return flow
    
 





    def link_info_upd(self, link): 
        #forward to another hub/join another Engress port
        link['status']+=1
        link['cur_hub']=link[['status','path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[0]), axis=1)
        link['cur_port']=link[['status','path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[1]), axis=1)
        return link
 
    
    
    def trace_upd(self, layer, i, sess, outputs, X,  my_result, my_link):
        """This module is to update my_result/my_link.
           Input:
             - layer:     which layer the device is located
             - i:         which position the device is located
             - sess:      trained deepQueueNet for inference
             - outputs:   delay
             - X:         features
             - my_result: shared result among GPUs  
             - my_link:   shared link traffic among GPUs  
        """
        #collect traffic to a device
        trace_in=self.IN['DataFrame{}'.format(i)].copy()  
        for c in range(20):
            _=my_link[(c,i)]  
            if _.shape[0]>0:
                trace_in=trace_in.append(_, ignore_index=True)
        trace_in=trace_in.sort_values('etime').reset_index(drop=True)
        #infer departure time  
        trace_out=self.infer(trace_in,
                             layer, 
                             sess,
                             outputs, X)
        #upd. my_link
        for _, __ in enumerate(self.PORTS[i]): 
            if  __ >-1: 
                eport_flow=trace_out[trace_out.cur_port==__]
                if eport_flow.shape[0]>0: 
                    my_link[(i,_)]=self.link_info_upd(eport_flow)
        #upd. my_result
        if layer=='edge':
            out_port0=my_result[(i-12)*2].append(trace_out[trace_out.cur_port==0][self.used_cols], ignore_index=True)
            out_port1=my_result[(i-12)*2+1].append(trace_out[trace_out.cur_port==1][self.used_cols], ignore_index=True)
            my_result[(i-12)*2]=out_port0
            my_result[(i-12)*2+1]=out_port1
            
            
       
        
 
        
                
    def run(self, gpu_number=1):
        tf.logging.set_verbosity(tf.logging.ERROR)
    
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
        self.flowsToport() 
        LINKS={(i,j): pd.DataFrame() for i in range(20) for j in range(20)} 
        
        
     
       
        with Manager() as MG:
            my_result=MG.dict() 
            my_link=MG.dict(LINKS)
            
            
            
            #deploy the task on multi-GPUs. Each GPU excutes the inference on a sub-graph.
            def pod_infer(device, my_result, my_link):
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(device) 
                if gpu_number==1:
                    core_bg, core_ed= 0,4
                    agg_bg, agg_ed= 4,12
                    edge_bg, edge_ed=12,20 
                    pc_bg, pc_ed=0,16 
                elif gpu_number==2:
                    core_bg, core_ed=((0,2), (2,4))[device]
                    agg_bg, agg_ed=((4,8), (8,12))[device]
                    edge_bg, edge_ed=((12,16), (16,20))[device]
                    pc_bg, pc_ed=((0,8), (8,16))[device]
                elif gpu_number==4:
                    core_bg, core_ed=((0,1), (1,2), (2,3), (3,4))[device]
                    agg_bg, agg_ed=((4,6), (6,8), (8,10), (10,12))[device]
                    edge_bg, edge_ed=((12,14), (14,16), (16,18), (18,20))[device]
                    pc_bg, pc_ed=((0,4), (4,8), (8,12), (12,16))[device]
                 
                 
                 
                
                with tf.Session() as sess:
                    saver = tf.train.import_meta_graph('{}/model.ckpt-{}.meta'.format(self.model_config.model, self.model_config.md))
                    saver.restore(sess, '{}/model.ckpt-{}'.format(self.model_config.model, self.model_config.md)) 
                    X = tf.get_collection('X')[0]   
                    outputs=tf.get_collection('outputs')[0]
                    #IRSA
                    for i in range(pc_bg, pc_ed): my_result[i]=pd.DataFrame(columns=self.used_cols)
                    for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, sess, outputs, X, my_result, my_link)
                    for i in range(pc_bg, pc_ed): my_result[i]=pd.DataFrame(columns=self.used_cols) #clear the unstable results
                    for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, sess, outputs, X, my_result, my_link)
                    for i in range(core_bg, core_ed): self.trace_upd('core', i, sess, outputs, X,my_result, my_link) 
                    for i in range(agg_bg, agg_ed): self.trace_upd('agg', i, sess, outputs, X, my_result, my_link)
                    for i in range(edge_bg, edge_ed): self.trace_upd('edge', i, sess, outputs, X, my_result, my_link)

                            
            #parallel reasoning                
            t0=time.time()         
            threads = []
            for i in range(gpu_number): 
                t=Process(target=pod_infer, args=(i, my_result, my_link))
                threads.append(t)
                t.start()
            for thr in threads:  thr.join()
                
                
            print("time used (total)", "%f min." %((time.time()-t0)/60))
            result=pd.DataFrame()
            for key in my_result.keys(): result=pd.concat([result, my_result[key]], ignore_index=True)
            result.reset_index(drop=True).to_csv('{}_pred.csv'.format(self.filename), index=False)
             
             
