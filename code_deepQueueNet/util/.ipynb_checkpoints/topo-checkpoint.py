"""
This script is to configure the fattree16.   
     1. flowsToport       traffic flow  
     2. get_src           forwarding
     3. addIngressFet     flow->features
     4. addEgressFet
     5. load_scaler       MinMaxScaler
     6. timeseries        features->batches of timeseries   
"""

import numpy as np 
from code_deepQueueNet.tools.trafficindicator import feaTure  #to create traffic indicators
from code_deepQueueNet.tools.MinMaxScaler import  load_scaler
 
 

 
    
    
class Init_TOPO:
    def __init__(self,flow, config, model_config):     
        self.PORTS=[[-1,-1,-1,-1,0,-1,1,-1,2,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1],
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
        self.config=config
        self.model_config=model_config
        self.flow=self.init_flow(flow) 

        
         
        
    def init_flow(self,flow):
        flow['dep_time']=flow['etime']
        flow['etime']=flow['timestamp (sec)']
        flow['cur_hub']=flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[0]))
        flow['cur_port']=flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
        flow['status']=0   #times the packet has been forwarded in the sys.
        return flow
    def flowsToport(self):
        self.IN=dict()
        for i in range(20): 
            self.IN['DataFrame{}'.format(i)]=self.flow[self.flow.src_hub==i].copy().sort_values('etime').reset_index(drop=True) 
        
            
    
    
     
    def load_scaler(self): 
        self.x_MIN, self.x_MAX, self.y_MIN, self.y_MAX, self.fet_cols, _=load_scaler(self.model_config.scaler) 
     
        
    def get_src(self, x): 
        src_port, status, path=x[0],x[1],x[2]
        if status==0: 
            return src_port 
        else: 
            last_hub=int(path.split('-')[status-1].split('_')[0])
            last_layer='core' if last_hub//4==0 else 'agg' if last_hub//12==0 else 'edge'
            cur_hub =int(path.split('-')[status].split('_')[0])
            cur_layer='core' if cur_hub//4==0 else 'agg' if cur_hub//12==0 else 'edge'
            if last_layer=='core':
                #downward forwarding 
                return 2+last_hub%2
            elif last_layer=='agg':
                if cur_layer=='core':
                    #upward forwarding
                    return (last_hub-4)//2
                else:
                    #downward forwarding 
                    return 2+last_hub%2
            else:
                #upward forwarding 
                return last_hub%2
            
           
    
        
     
    def addIngressFet(self, dt): 
        dt['inter_arr_sys']=dt['timestamp (sec)'].diff()
    def addEgressFet(self, dt): 
        ins=feaTure(dt, 
                    self.config.no_of_port,  
                    self.config.no_of_buffer, 
                    self.config.window, 
                    self.config.ser_rate)
        Count_dst_SET, LOAD=ins.getCount()
        for i in range(self.config.no_of_port): dt['TI%i' %i]=LOAD[i] 
        for i in range(self.config.no_of_port):
            for j in range(self.config.no_of_buffer):
                dt['load_dst%i_%i' %(i,j)]=np.array(Count_dst_SET[(i,j)])
           
               
     
    
    def timeseries(self, sample):
        #total number of time-series samples would be x.shape[0]-timesteps+1
        dim0=sample.shape[0]-self.config.TIME_STEPS+1
        dim1=sample.shape[1]
        x=np.zeros((dim0, self.config.TIME_STEPS, dim1))
        for i in range(dim0): 
            x[i]=sample[i:self.config.TIME_STEPS+i]
        return x