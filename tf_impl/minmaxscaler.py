import numpy as np
import pandas as pd
 
    
    
    
    
    
class cScaler:
    def __init__(self, MIN, MAX, name='x', keys=None, no_of_port=None, no_of_buffer=None):
        self.MIN=MIN
        self.MAX=MAX
        self.name=name
        self.keys=keys
        self.no_of_port=no_of_port
        self.no_of_buffer=no_of_buffer
    
    def cluster(self):
        """Combine scaler of columns within the same category"""
        for key in self.keys:
            if 'port_load' in key:
                self.MIN[key]=f('port_load', self.MIN, self.no_of_port, None, 'min')
                self.MAX[key]=f('port_load', self.MAX, self.no_of_port, None, 'max')  
            elif 'load' in key:
                self.MIN[key]=f('load_dst', self.MIN, self.no_of_port, self.no_of_buffer, 'min')
                self.MAX[key]=f('load_dst', self.MAX, self.no_of_port, self.no_of_buffer, 'max')  
            elif 'inter_arr' in key: 
                self.MIN[key]=f('inter_arr', self.MIN, self.no_of_buffer, None, 'min')
                self.MAX[key]=f('inter_arr', self.MAX, self.no_of_buffer, None, 'max')
            elif 'weight' in key:  
                self.MIN[key]=f('weight', self.MIN, self.no_of_buffer, None, 'min')
                self.MAX[key]=f('weight', self.MAX, self.no_of_buffer, None, 'max')
            else:
                pass
            
    def save(self, folder):
        self.MIN.to_csv(folder+'/{}_MIN.csv'.format(self.name), header=None)
        self.MAX.to_csv(folder+'/{}_MAX.csv'.format(self.name), header=None)
        
        
        
    
def load_scaler(folder):
    x_MIN=pd.read_csv(folder+'/x_MIN.csv', names=['key','value']).set_index('key')['value']
    x_MAX=pd.read_csv(folder+'/x_MAX.csv', names=['key','value']).set_index('key')['value']
    y_MIN=pd.read_csv(folder+'/y_MIN.csv', names=['key','value']).set_index('key')['value']
    y_MAX=pd.read_csv(folder+'/y_MAX.csv', names=['key','value']).set_index('key')['value']
    fet_cols=list(x_MIN.keys())
    target=list(y_MIN.keys())
    return x_MIN, x_MAX, y_MIN, y_MAX, fet_cols, target
  
 
 



def f(key, series, n1, n2=None, op='max'):
    if n2:
        keys=['{}{}_{}'.format(key, i, j) for i in range(n1) for j in range(n2)]
    else:
        keys=['{}_sys'.format(key)]+['{}{}'.format(key, i) for i in range(n1)]   
    keys=list(filter(lambda x: x in series.index, keys))
    return series[keys].max() if op=='max' else series[keys].min()
    

 