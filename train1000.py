import os
import sys
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from matplotlib import pyplot as plt
import copy
import glob
import pickle


from termcolor import cprint
import pprint

from model import DeviceModel
from Dataset import H5Dataset

pytorch_seed = 0
torch.manual_seed(pytorch_seed)


###### trained device config #######
# port_rate = 10 * 1024 * 1024 # byte
# port_rate_max = 1024 * 1024 * 1024 * 1024
# mean_pkt_size = 1000.0
###################################


input_dim = 12 
embed_dim = 200
hidden_dim = 100
output_dim = 1

batch_size = 64 * 2 * 2
seq_len = 42 

batch_num_per_epoch = 2000

use_gpu = True
max_epoch = 1000


lr = 1e-3
weight_decay = 1e-3  #  for SGD
momentum = 0.9 # for SGD

l2_reg_lambda = 1e-1

input_gap = 1

#### training config ####################

identifier = "default"


save_base_dir = f"saved/{identifier}"

model_dir = "{}/saved_model".format(save_base_dir)
saved_model_name = "best_model.pt"

saved_loss_figure_name = "{}_train_eval_loss.png".format(identifier)
saved_eval_loss_figure_name = "{}_eval_loss.png".format(identifier)
saved_train_loss_figure_name = "{}_train_loss.png".format(identifier)


plot_dir = "{}/train_plot".format(save_base_dir)

#########################################
base_dir = "./DeepQueueNet-synthetic-data/data"




def save_model(model, out_file):
    torch.save(model.state_dict(), model_dir + "/" + out_file)


def train():

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print("Mkdir {}".format(plot_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("Mkdir {}".format(model_dir))

    train_dataset = H5Dataset(base_dir, mode="train")
    valid_dataset = H5Dataset(base_dir, mode="sampled_valid")

    # dataset_indices = list(range(len(train_dataset)))
    
    print("Train data len:", len(train_dataset))
    print("Valid data len:", len(valid_dataset))

    # model = DeviceModelBiLSTM(seq_len, input_dim, embed_dim, hidden_dim, output_dim)
    model = DeviceModel(seq_len, input_dim, embed_dim, hidden_dim, output_dim)
    model = model.cuda()

    dataloader = DataLoader(train_dataset, batch_size = batch_size,
                            shuffle = True,
                             num_workers = 0, 
                            # pin_memory = True,
                            # sampler = SubsetRandomSampler(sample(dataset_indices, batch_size * batch_num_per_epoch))
                            )
        
    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr=lr,
    #                             momentum = momentum,
    #                             weight_decay = weight_decay
    #                             )
    optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr = lr
                                )

    loss_func = torch.nn.MSELoss()
    eval_loss_func = copy.deepcopy(loss_func)

    # print("input file:", input_file)
    print("input dim:", input_dim)
    print("batch size:", batch_size)
    print("seed:", pytorch_seed)
    print("lr:", lr)
    print("optimizer:", optimizer.__class__.__name__)
    print("loss func:", loss_func.__class__.__name__) 

    epoch_batch_num = len(dataloader)

    train_epoch_losses = []
    eval_epoch_losses = []
    total_batch_num = 0
    iter_num = 0

    cached_loss = "Init"
    sum_of_loss = 0
    prev_time = time.time()
    
    for epoch in range(max_epoch):

        ############## train part ###################

        print("\nepoch:", epoch, "iter:", iter_num)
        batch_num = 0

        model.train()

        for idx, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()

            batch_num += 1
            total_batch_num += 1
            out = model(batch_x)

            loss = loss_func(out, batch_y)
            prev_loss = loss.item()

            # decoder_fc_out = model.decoder.fc_out.parameters()
            # for para in  decoder_fc_out: # regularizer for last feedforward layer of decoder
            #     loss +=  l2_reg_lambda * torch.sum(para.pow(2)) # L2 regularization

            loss.backward()
            optimizer.step()

            sum_of_loss += prev_loss
            # print the batch num and loss value (per 100 batch)
            
            if batch_num % 100 == 0:
                cached_loss = float(sum_of_loss / batch_num)
            print("\r", end="")
            print("batch: {} / {} avg loss: {}".format(idx + 1, epoch_batch_num, cached_loss), end="") 

            ######## end of train part ############

            if total_batch_num % 1000 == 0:
                iter_num += 1
                print()
                train_epoch_avg_loss = sum_of_loss / batch_num
                train_epoch_losses.append(train_epoch_avg_loss)
                cprint("iter avg loss: {}\n".format(train_epoch_avg_loss), "yellow")
                # print(input_dim, batch_size, pytorch_seed, optimizer.__class__.__name__, loss_func.__class__.__name__)   
                print()
                cached_loss = "Init"
                sum_of_loss = 0
                batch_num = 0

                ############# validation part ##########
                print("Iter num:", iter_num)
                valid(model, eval_loss_func, validation_loader, train_epoch_losses, eval_epoch_losses)
                ############## end of validation part ######
                print("[===] Time used: %.2f min." % ((time.time() - prev_time) / 60))
                prev_time = time.time()

    # print training settings
    print()
    print("input dim:", input_dim)
    print("batch size:", batch_size)
    print("seed:", pytorch_seed)
    print("optimizer:", optimizer.__class__.__name__)
    print("loss func:", loss_func.__class__.__name__)    

def valid(model, eval_loss_func, validation_loader, train_epoch_losses, eval_epoch_losses):
    model.eval()

    loss_list = []
    eval_loss = 0.0
    with torch.no_grad():
        eval_epoch_batch_num = len(validation_loader)

        for idx, (batch_x, batch_y) in enumerate(validation_loader):
            
            out = model(batch_x)
            loss = eval_loss_func(out, batch_y)

            eval_loss += loss.item()
            loss_list.append(loss.item())
            print("\r", end="")
            print("Eval batch: {} / {}".format(idx, eval_epoch_batch_num), end="")
        
        print()
        epoch_avg_loss = eval_loss / eval_epoch_batch_num
        cprint("Eval avg loss: {} ".format(epoch_avg_loss), "blue", end = "")
        eval_epoch_losses.append(epoch_avg_loss)
        print()

        ### save the model for the epoch with best eval loss
        if epoch_avg_loss == min(eval_epoch_losses):
            save_model(model, saved_model_name)
            cprint("Best model saved!", "red")
            # print("Model as:", saved_model_name)
    
    model.train()

    eval_epoch_losses_array = np.array(eval_epoch_losses)
    train_epoch_losses_array = np.array(train_epoch_losses)
    epoch_list = np.array(list(range( len(eval_epoch_losses) ) ) )

    plt.plot(epoch_list, eval_epoch_losses_array, color = "red", label="eval")
    plt.plot(epoch_list, train_epoch_losses_array, color = "blue", label="train")
    plt.xlabel('iter')
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("./{}/{}".format(plot_dir, saved_loss_figure_name))
    plt.clf()


    plt.plot(epoch_list, eval_epoch_losses_array, color = "red")
    plt.xlabel('iter')
    plt.ylabel("loss")
    plt.savefig("./{}/{}".format(plot_dir, saved_eval_loss_figure_name))
    plt.clf()

    plt.plot(epoch_list, train_epoch_losses_array, color = "blue")
    plt.xlabel('iter')
    plt.ylabel("loss")
    plt.savefig("./{}/{}".format(plot_dir, saved_train_loss_figure_name))
    plt.clf()

    # cprint(f"PLOTTED in dir: {plot_dir}\n", "yellow")
    print("Identifier:", identifier, "\n")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')
    train()
    print("FINISHED ...")
