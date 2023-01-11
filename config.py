import pandas as pd
import numpy as np



class BaseConfig:
    test_size=0.2 #train_test_split ratio
    sub_rt=0.005  #subsampling for Eval.
    TIME_STEPS=42
    BATCH_SIZE=32*8
    modelname='4-port switch/FIFO'
    no_of_port=4
    no_of_buffer=1
    ser_rate=2.5*1024**2
    sp_wgt=0.
    seed=0
    window=63   #window size to cal. average service time.
    no_process=8 #multi-processing:no of processes used. 
    epochs=6
    n_outputs=1 
    learning_rate=0.001
    l2=0.1
    lstm_params={'layer':2,   'cell_neurons':[200,100],     'keep_prob':1}  
    att=64. #attention output layer dim
    mul_head=3
    mul_head_output_nodes=32



class ModelConfig:
    scaler='./data/_scaler'  
    model='./trained/model' 
    md=341
    # train_sample= './trained/sample/train.h5'
    # test1_sample= './trained/sample/test1.h5'
    # test2_sample= './trained/sample/test2.h5'
    bins=100
    errorbins='./data/_error'
    error_correction=False
    
    
