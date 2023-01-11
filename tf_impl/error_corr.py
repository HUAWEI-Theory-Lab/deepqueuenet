# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache-2.0 License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache-2.0 License for more details.

import pandas as pd 
import numpy as np 




 
    
def clf(x, bins): 
    locs=np.where(x<=bins)[0]
    if len(locs)==0:
        return len(bins)-2
    else: 
        return max(0, locs[0]-1)


def get_error_dis(config, model_config):
    ERR=dict()
    BINS=dict()
        
    df=pd.read_csv('{}/train.csv'.format(model_config.errorbins))
    for i in range(config.no_of_buffer):
        ts=df[df['priority']==i].copy()
        bins=np.linspace(ts.prediction.min(), ts.prediction.max(), model_config.bins+1)
        ts['bins'] =ts['prediction'].apply(lambda x: clf(x, bins)) 
        err=ts.groupby('bins')['error'].apply(lambda x: x.values)

        #make up for missing bins
        mk=pd.DataFrame(index=range(model_config.bins))
        mk.index.name='bins'
        mk=mk.join(pd.DataFrame(err)).fillna('')
        ERR[i]=mk['error']
        BINS[i]=bins
    return ERR, BINS



def error_correction(df, config, ERR, BINS): 
    for i in range(config.no_of_buffer):
        t=df[df['priority']==i].copy()
        t['bins']=t['delay'].apply(lambda x: clf(x, BINS[i]))
        t['delay']=t[['delay','bins']].apply(lambda x: x[0]-np.random.choice(ERR[i][x[1]]) if len(ERR[i][x[1]])>0 else x[0], axis=1)

        if i==0:
            ts=t
        else:
            ts=pd.concat([ts,t], ignore_index=True)
    ts.sort_values('etime', inplace=True)
    return ts