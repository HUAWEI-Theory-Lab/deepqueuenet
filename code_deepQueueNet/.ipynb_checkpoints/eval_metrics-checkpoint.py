"""
This script is to load trained deviceModel for evaluation.   
"""
from code_deepQueueNet.featureET import trace2Samples
import tensorflow as tf 
import h5py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as measures
from scipy import stats
import warnings
warnings.filterwarnings("ignore") 
 

    
    
    
    
class REPO(trace2Samples):
    def __init__(self, config, model_config, target):
        super(REPO, self).__init__(config, target)
        self.model_config=model_config
 
        
        
    def loadModel_and_Eval(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('{}/model.ckpt-{}.meta'.format(self.model_config.model, self.model_config.md))
            saver.restore(sess, '{}/model.ckpt-{}'.format(self.model_config.model, self.model_config.md)) 
            X = tf.get_collection('X')[0]   
            outputs=tf.get_collection('outputs')[0]
            
            x,y  =self.load_sample('train', self.model_config.train_sample)
            x1,y1=self.load_sample('test1', self.model_config.test1_sample)
            x2,y2=self.load_sample('test2', self.model_config.test2_sample)
            y_pred =sess.run(outputs, feed_dict={X: x}).flatten()
            y1_pred=sess.run(outputs, feed_dict={X: x1}).flatten()
            y2_pred=sess.run(outputs, feed_dict={X: x2}).flatten()
            
            
            self.load_scaler()
            y_MAX=self.y_MAX['time_in_sys']
            y_MIN=self.y_MIN['time_in_sys']
            y_RANGE=y_MAX-y_MIN
            self.y =y[:,0]*y_RANGE+y_MIN 
            self.y1=y1[:,0]*y_RANGE+y_MIN 
            self.y2=y2[:,0]*y_RANGE+y_MIN 
            self.y_pred =y_pred*y_RANGE+y_MIN 
            self.y1_pred=y1_pred*y_RANGE+y_MIN 
            self.y2_pred=y2_pred*y_RANGE+y_MIN 
            
            
    def learning_curve(self):
        mse_txt=pd.read_csv('{}/acc.txt'.format(self.model_config.model), header=None)
        
        """
        train/test_endogeny: MAP  
        test_exogenesis: MAP+Poisson
        """
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(111)
        mse_train=[float(f.split(':')[-1]) for f in mse_txt[1].values]
        mse_test1=[float(f.split(':')[-1]) for f in mse_txt[2].values]
        mse_test2=[float(f.split(':')[-1]) for f in mse_txt[3].values]
        lns1=ax1.plot([(1+i)*1000 for i in range(len(mse_train))], mse_train, label='MSE: train', linewidth=2.)
        lns2=ax1.plot([(1+i)*1000 for i in range(len(mse_test1))], mse_test1, 'C1', label='MSE: test_endogeny', linewidth=2.)
        ax1.set_xlabel('training step', fontsize = 14)
        ax1.set_ylabel('MSE: train/test_endogeny', fontsize = 14)
        ax2 = ax1.twinx()   
        lns3=ax2.plot([(1+i)*1000 for i in range(len(mse_test2))], mse_test2,'C2', label='MSE: test_exogenesis', linewidth=2.)
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, fontsize = 12)
        ax2.set_ylabel('MSE: test_exogenesis', fontsize = 14)
        plt.tick_params(labelsize=12)
        plt.show();  
        
        
        
    def regression_rho(self):
        coll={'train': [131, self.y*1e3, self.y_pred*1e3], 
              'test: endogeny': [132, self.y1*1e3, self.y1_pred*1e3],
              'test: exogenesis': [133, self.y2*1e3, self.y2_pred*1e3]}
        

        plt.figure(figsize=(20,5))
        for k, v in coll.items():
            Max =max(max(v[1]),  max(v[2]))
            plt.subplot(v[0])
            plt.scatter(v[1], v[2], alpha=0.6, label='prediction (rho={})'.format(measures.pearsonr(v[1], v[2])[0]))
            plt.plot([0,Max],[0,Max], color='k',label='regression line')
            plt.legend()
            plt.xlabel('actual delay (ms)')
            plt.ylabel('prediction');
            plt.title('\n\n{}'.format(k))
        plt.suptitle('regression plot\n\n\n\n\n');
        
        
        
    def pdf_cdf(self, df, disp):
        plt.figure(figsize=(16,5))
        plt.subplot(121)
        bins=np.histogram(np.hstack((df['delay'].values, df['prediction'].values)), bins=100)[1]  
        plt.hist(df['delay'].values, bins, density=False,  histtype='step', label='delay', lw=1.5);
        plt.hist(df['prediction'].values, bins, density=False,  histtype='step', label='prediction',linestyle=':',lw=1.5);
        plt.title('pdf')
        plt.legend(frameon=False, loc="best")

        plt.subplot(122)
        _res=stats.relfreq(df['delay'].values, numbins=100)
        res_=stats.relfreq(df['prediction'].values, numbins=100)
        _x=_res.lowerlimit + np.linspace(0, _res.binsize*_res.frequency.size, _res.frequency.size)
        _y=np.cumsum(_res.frequency)
        x_=res_.lowerlimit + np.linspace(0, res_.binsize*res_.frequency.size, res_.frequency.size)
        y_=np.cumsum(res_.frequency)
        plt.plot(_x, _y,label='delay',lw=1.5)
        plt.plot(x_, y_,  linestyle=':', label='prediction',lw=1.5)
        plt.title('cdf');
        plt.legend(frameon=False, loc="best")
        plt.suptitle('delay analysis: {}'.format(disp))
    
    
    
    
    
    def distrib(self):
        Result=pd.DataFrame(self.y, columns=['delay'])
        Result['prediction']=self.y_pred
        Result['error']=self.y_pred-self.y
        Result['priority']=0
        Result.to_csv('./data/{}/_error/train.csv'.format(self.config.modelname), index=False)
        Result1=pd.DataFrame(self.y1, columns=['delay'])
        Result1['prediction']=self.y1_pred
        Result2=pd.DataFrame(self.y2, columns=['delay'])
        Result2['prediction']=self.y2_pred
        
        
        self.pdf_cdf(Result*1e3, disp='train') 
        self.pdf_cdf(Result1*1e3, disp='test_endogeny')   
        self.pdf_cdf(Result2*1e3, disp='test_exogenesis')           

        