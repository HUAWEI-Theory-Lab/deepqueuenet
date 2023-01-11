import numpy as np 
 

 
        
class  build_timeseries:
    def __init__(self, sample, target_col_index=[-1]):
        # target_col_index is the index of column that would act as output column
        self.sample=sample
        self.target_col_index=[sample.shape[1]+i if i<0 else i for i in target_col_index]
        
    
    def timeseries(self, TIME_STEPS): 
        # total number of time-series samples would be sample.shape[0] - TIME_STEPS+1
        dim_0=self.sample.shape[0]-TIME_STEPS+1
        dim_1=self.sample.shape[1]-len(self.target_col_index)  #number of features 
        x=np.zeros((dim_0, TIME_STEPS, dim_1))
        y=np.zeros((dim_0, len(self.target_col_index)))
        
        sample_x=np.delete(self.sample, self.target_col_index, 1)
        sample_y=self.sample[:, self.target_col_index]
        
        for i in range(dim_0): 
            x[i]=sample_x[i:TIME_STEPS+i]
            y[i]=sample_y[TIME_STEPS+i-1]
        return x, y     
         
     
    




    
 