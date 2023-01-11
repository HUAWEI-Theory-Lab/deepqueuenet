# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache-2.0 License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache-2.0 License for more details.

import numpy as np 
 


    
    
    
class feaTure:
    def __init__(self, inflow, no_of_port, no_of_buffer, WINDOW, ser_rate):
        self.inflow=inflow
        self.no_of_port=no_of_port 
        self.no_of_buffer=no_of_buffer
        self.tau_list=(inflow['pkt len (byte)'].rolling(WINDOW, min_periods=1).mean()/ser_rate).values    
    
    def getNum(self, Time, i, TAU):
        t=Time[i] #current time 
        no=0
        ix=i-1
        while ix>=0:
            if t-Time[ix]<=TAU:
                no+=1
                ix-=1
            else:
                break 
        return no 
 
 
    
    def getCount(self):
        Time=self.inflow['timestamp (sec)'].values
        SRC=self.inflow['src'].values
        DST=self.inflow['dst'].values
        Prio=self.inflow['priority'].values
        C_dst_SET={(i,j):[0] for i in range(self.no_of_port) for j in range(self.no_of_buffer)}
        LOAD={i: [0.] for i in range(self.no_of_port)}
     
        
        
        
        for t in range(1, len(Time)):
            TAU=self.tau_list[t]  #adapted service time to cal. the corresponding work loads. 
            ix=self.getNum(Time, t, TAU)
            src=SRC[t-ix:t] if ix>0 else np.array([])
            dst=DST[t-ix:t] if ix>0 else np.array([])
            prio=Prio[t-ix:t] if ix>0 else np.array([])

            for i in range(self.no_of_port):
                LOAD[i].append(np.sum(src==i))
                for j in range(self.no_of_buffer):
                    C_dst_SET[(i,j)].append(np.sum((dst==i) & (prio==j)))  


        for i in range(self.no_of_port):
            LOAD[i]=np.mean(np.array(LOAD[i])[SRC==i])  if len(np.array(LOAD[i])[SRC==i])>0 else 0.
  
        return C_dst_SET, LOAD
        
    
    
    
     



   
     
    
     



