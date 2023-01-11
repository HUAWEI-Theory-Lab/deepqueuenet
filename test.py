from random import shuffle
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import glob
import os

import time
import sys

from model import DeviceModel
from Dataset import DeviceDataset, H5Dataset
import seaborn as sns

import scipy.stats as measures
from scipy.stats import wasserstein_distance

import pickle

pytorch_seed = 0
torch.manual_seed(pytorch_seed)

input_dim = 12

embed_dim = 200
hidden_dim = 100
output_dim = 1

batch_size = 64 * 2 * 2
seq_len = 42

use_gpu = True

identifier = "default"

saved_figure_name = "gt_pred_cmp_fig.png"

saved_cdf_figure_name = "cdf_fig.png"

saved_ccdf_figure_name = "ccdf_fig.png"

saved_pdf_figure_name = "pdf_fig.png"


base_dir = "./DeepQueueNet-synthetic-data/data"

plot_dir = "saved/{}/test_plot".format(identifier)
test_saved_model = "saved/{}/saved_model/best_model.pt".format(identifier)

def ideal_delay_dist(lambda_gen_rate, miu):
        '''
        lambda is the generation rate while 1/lambda =  mean of packet interval time

        1 / miu = mean of packet size / port_rate (the mean processing time for each packet) 

        '''
        # By Little's formula, we have W = 1 / (miu - lambda) , ref to https://www.comp.nus.edu.sg/~cs3260/MM1.pdf 
        W = 1 / (miu - lambda_gen_rate)
        # W is the average delay
        print("miu", miu)
        print("lambda_gen_rate", lambda_gen_rate)
        # Thus, the distribution of delay is exponential(1/W)
        lmbd = (1 / W)
        print("lmbd", lmbd)
        print()
        return lambda x: lmbd * np.exp( - lmbd * x) # the expected PDF of delay

# def inv_max_min_normalize(col, col_max, col_min, epsilon = 1e-16):
#     return col * (col_max - col_min + epsilon) + col_min

def test():
    if not os.path.exists(test_saved_model):
        print("Test model not exists!", file=sys.stderr)

    if not os.path.exists(plot_dir):
        print("Mkdir ", plot_dir)
        os.makedirs(plot_dir)
    
    print("Tested model:", test_saved_model)

    dataset = H5Dataset(base_dir, mode="test")
    # dataset.test_run()
    test_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)


    model = DeviceModel(seq_len, input_dim, embed_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(test_saved_model))
    model = model.cuda()
    model.eval()

    test_loss_func = torch.nn.MSELoss()

    prediction_list = []
    ground_truth_list = []

    print("Test start ...")
    with torch.no_grad():
        test_loss = 0.0
        test_epoch_batch_num = len(test_dataloader) 
        start_time = time.time()
        for idx, (batch_x, batch_y) in enumerate(test_dataloader):
            
            out = model(batch_x)
            loss = test_loss_func(out, batch_y)

            prediction_list.append(out.cpu().detach().numpy())
            ground_truth_list.append(batch_y.cpu().detach().numpy())

            test_loss += loss.item()
            print("\r", end="")
            print("Test batch: {} / {}".format(idx, test_epoch_batch_num), end="")
        
        print()
        used_time = time.time() - start_time
        print("Total time:", used_time)
        print("Avg batch time:", used_time / test_epoch_batch_num)
        print("Avg pkt time:", used_time / len(dataset))
    print("Test done ...")

    
    pred_org = np.concatenate(np.array(prediction_list)).flatten()
    pred = np.sort(pred_org)
    gt_org = np.concatenate(np.array(ground_truth_list)).flatten()
    gt = np.sort(gt_org)

    print("Data len:", len(pred_org))
    # print(pred_org.shape)
    # sys.exit(0)
    
    coef = np.corrcoef(gt_org, pred_org)[1,0]
    # coef = np.corrcoef(gt, pred)[1,0]
    print("coeff done ...")

    empty_array = np.zeros((len(gt),))
    w1_dist = wasserstein_distance(gt_org, pred_org) / wasserstein_distance(gt_org, empty_array)
    print("W1 distance done ...")

    # plt.plot(gt, pred, color = "blue", linestyle = "-")
    # plt.plot(gt, gt, color = "gray", linestyle = "-.")
    max_pos = max(max(pred_org), max(gt_org))
    plt.scatter(gt_org, pred_org, s = 1)
    plt.plot([0,max_pos],[0,max_pos], linestyle = "-.", color = "red")
    plt.xlabel("GT")
    plt.ylabel("prediction result")
    plt.title("$\\rho  = {}$,  W1 = {}".format(coef, w1_dist))

    plt.savefig("{}/{}".format(plot_dir, saved_figure_name))

    plt.clf()

    ####### PLOT CDF
    pr = np.cumsum(np.ones_like(pred))
    pr = pr / len(pr)

    plt.plot(pred, pr, color = "red", linestyle = "-")
    plt.plot(gt, pr, color = "blue", linestyle = "-." )
    plt.xlabel("Delay")
    plt.ylabel("Pr")
    plt.title("CDF")
    plt.savefig("{}/{}".format(plot_dir, saved_cdf_figure_name))


    ##### PLOT CCDF 
    ccdf = np.log(1 - pr + 1e-20)
    x_axis_max = max(max(pred), max(gt))
    plt.plot(pred, ccdf, color = "red", linestyle = "-")
    plt.plot(gt, ccdf, color = "blue", linestyle = "-." )
    plt.xlabel("Delay")
    plt.ylabel("log(1 - CDF)")
    plt.title("CCDF")
    plt.axis((0,x_axis_max,-10,0))
    plt.savefig("{}/{}".format(plot_dir, saved_ccdf_figure_name))


    plt.clf()

    ###### PLOT PDF

    # gt_hist, gt_bins = np.histogram(gt_org, bins = 1000)
    # pred_hist, pred_bins = np.histogram(pred_org)
    # hist = gt_hist
    # bins = gt_bins
    # bin_centers = (bins[1:]+bins[:-1])*0.5
    # plt.plot(bin_centers, hist)

    bins = 100
    sns.distplot(gt_org, 
                hist=True, 
                kde=True, 
                bins=bins, 
                color = 'blue',
                label = "GT",
                # hist_kws={'edgecolor':'black'}
                )
    sns.distplot(pred_org, 
                hist=True, 
                kde=True, 
                bins=bins, 
                color = 'red',
                label = "Predction",
                # hist_kws={'edgecolor':'black'}
                )
    plt.legend()
    plt.title("$\\rho  = {}$".format(coef))

    plt.savefig("{}/{}".format(plot_dir, saved_pdf_figure_name))


if __name__ == '__main__':

    test()


    


