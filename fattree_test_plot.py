
import os, sys

import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('ticks')


import tqdm
import matplotlib.patches as mpatches
import scipy.stats as measures

identifier = "default"

base_dir = "./DeepQueueNet-synthetic-data/data"
plot_dir = "saved/{}/fattree_plot".format(identifier)


if not os.path.exists("saved/{}".format(identifier)):
    print("Invalid identifier!")
    sys.exit(0)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def showFig(df):
    for col in ['delay', 'jitter']:
        fig,(ax) = plt.subplots(figsize=(8,5))
        a_val = 1.
        colors = ['r','b','g']
        circ1 = mpatches.Patch(edgecolor=colors[0],alpha=a_val,linestyle ='-',label='MAP - sim', fill=False)
        circ2 = mpatches.Patch(edgecolor=colors[0],alpha=a_val,linestyle ='--',label='MAP - prediction',fill=False)
        circ3 = mpatches.Patch(edgecolor=colors[1],alpha=a_val,linestyle ='-',label='Poisson - sim', fill=False)
        circ4 = mpatches.Patch(edgecolor=colors[1],alpha=a_val,linestyle='--', label='Poisson - prediction', fill=False)
        circ5 = mpatches.Patch(edgecolor=colors[2],alpha=a_val,linestyle ='-',label='Onoff - sim', fill=False)
        circ6 = mpatches.Patch(edgecolor=colors[2],alpha=a_val,linestyle='--', label='Onoff - prediction', fill=False)
        
        
        for i, c in zip(['MAP','Poisson','Onoff'], colors):
            bins=np.histogram(np.hstack((df[df.tp==i][col+'_sim'].values, 
                                         df[df.tp==i][col+'_pred'].values)), bins=100)[1]  
            plt.hist(df[df.tp==i][col+'_sim'].values, bins, density=True, color=c, histtype='step',  linewidth=1.5)
            plt.hist(df[df.tp==i][col+'_pred'].values, bins, density=True, color=c, histtype='step', linestyle='--', linewidth=1.5)
        ax.legend(handles = [circ1,circ2,circ3,circ4,circ5,circ6],loc=1, fontsize = 14)
        plt.xlabel(col.capitalize()+' (sec)', fontsize = 14)
        plt.ylabel('PDF', fontsize = 14)
        plt.tick_params(labelsize=12)

        plt.savefig("{}/fattree_pdf_{}.png".format(plot_dir, col))
        
        
        fig,(ax) = plt.subplots(figsize=(8,5))
        for i, c in zip(['MAP','Poisson','Onoff'], colors):
            res=stats.relfreq(df[df.tp==i][col+'_sim'].values, numbins=100)
            x=res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)
            y=np.cumsum(res.frequency)
            plt.plot(x, y,color=c,  linewidth=1.5)
            res=stats.relfreq(df[df.tp==i][col+'_pred'].values, numbins=100)
            x=res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)
            y=np.cumsum(res.frequency)
            plt.plot(x, y,color=c, linestyle='--',   linewidth=1.5)
        plt.xlabel(col.capitalize()+' (sec)', fontsize = 14)
        plt.ylabel('CDF', fontsize = 14)
        ax.legend(handles = [circ1,circ2,circ3,circ4,circ5,circ6],loc=4, fontsize = 14)
        plt.tick_params(labelsize=12)

        plt.savefig("{}/fattree_cdf_{}.png".format(plot_dir, col))
         
def plot_regression(pred_org, gt_org, saved_figure_name):
    max_pos = max(max(pred_org), max(gt_org))
    plt.scatter(gt_org, pred_org, s = [10 for _ in range(len(gt_org))], alpha = 0.5)
    plt.plot([0,max_pos],[0,max_pos], linestyle = "-.", color = "red")
    plt.xlabel("GT")
    plt.ylabel("prediction result")
    # plt.title("$\\rho  = {}$,  W1 = {}".format(coef, w1_dist))

    plt.savefig("{}/{}".format(plot_dir, saved_figure_name))
    plt.clf()

def plot_all_regression(df):
    for col in ["delay", "jitter"]:
        for tp in ['MAP','Poisson','Onoff']:
            gt = df[df.tp==tp][col+'_sim'].values
            pred = df[df.tp==tp][col+'_pred'].values
            saved_name = "fattree_regression_{}_{}.png".format(tp, col)
            plot_regression(pred, gt, saved_name)


def plot_avgjitter_regression(df):

    for tp in ['MAP','Poisson','Onoff']:
        df_mean = df[df.tp==tp].groupby(["src_port", "path"])[["jitter_pred", "jitter_sim"]].mean()
        gt = df_mean["jitter_sim"].values
        pred = df_mean["jitter_pred"].values
        saved_name = "fattree_regression_{}_avgjitter.png".format(tp)
        plot_regression(pred, gt, saved_name)




        
def mergeTrace():
    result=pd.DataFrame()
    for traffic_pattern in tqdm.tqdm(['onoff','poisson','map']):
        for filename in  ['rsim1', 'rsim2', 'rsim3', 'rsim4', 'rsim5']:
            t=pd.read_csv('{}/fattree16/{}/{}_pred.csv'.format(base_dir, traffic_pattern, filename))
            t['delay_sim']=t['dep_time']-t['timestamp (sec)']
            t['delay_pred']=t['etime']-t['timestamp (sec)']
            t['fd']=t['path'].apply(lambda x: len(x.split('-'))) 
            t['jitter_sim']=t.groupby(['src_port', 'path'])['delay_sim'].diff().abs()
            t['jitter_pred']=t.groupby(['src_port', 'path'])['delay_pred'].diff().abs()
            if traffic_pattern=='map':
                t['tp']='MAP'
            else:
                t['tp']=traffic_pattern.capitalize()
            result=pd.concat([result, t], ignore_index=True)     
    return result 
             




def show_result():
    traffic_pattern = "map"
    filename = "rsim1"

    t = pd.read_csv('{}/fattree16/{}/{}_pred.csv'.format(base_dir, traffic_pattern, filename))
    t['delay_sim']=t['dep_time']-t['timestamp (sec)']
    t['delay_pred']=t['etime']-t['timestamp (sec)']
    t['fd']=t['path'].apply(lambda x: len(x.split('-'))) 
    t['jitter_sim']=t.groupby(['src_port', 'path'])['delay_sim'].diff().abs()
    t['jitter_pred']=t.groupby(['src_port', 'path'])['delay_pred'].diff().abs()
    t["delay_diff"] = t["delay_sim"] - t["delay_pred"]
    t["delay_diff_rate"] = t["delay_diff"] / t["delay_sim"]

    print(t[["delay_sim", "delay_pred", "delay_diff", "delay_diff_rate", "path"]].iloc[:50])
    print("mean    std")
    print("sim")
    print(t["delay_sim"].mean(), t["delay_sim"].std())
    print("pred")
    print(t["delay_pred"].mean(), t["delay_pred"].std())

    def cmp_diff(x, y, num = 10):
        x = x.sort_values(ascending = True).values
        y = y.sort_values(ascending = True).values
        x_vals, y_vals = [], []
        x_len, y_len = len(x), len(y)
        for idx in range(1, num):
            x_val, y_val = x[int(idx / num * x_len)], y[int(idx / num * y_len)]
            x_vals.append(x_val)
            y_vals.append(y_val)
        
        print("Pos\t\tx\t\ty\t\tx-y")
        for idx, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
            print("{}/{}\t\t{:.8f}\t{:.8f}\t{:.8f}".format(idx+1, num, x_val, y_val, x_val - y_val))
    
    cmp_diff(t["delay_sim"], t["delay_pred"])
        





if __name__ == "__main__":
    # show_result()

    result = mergeTrace().dropna()
    # plot_all_regression(result)
    plot_avgjitter_regression(result)

    showFig(result)





