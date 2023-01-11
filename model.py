
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

import numpy as np

import random
import math
import time

from SelfAttention import ScaledDotProductAttention


class Encoder(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, hidden_dim, n_layers = 2, dropout = 0):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding  = nn.Linear(input_dim, emb_dim)
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout = dropout, bidirectional = True, batch_first = True)
        
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, seq):
        
        # seq : [batch size , seq len, input_dim]
        embedded = self.embedding(seq)
        # embedded = self.dropout(embedded)
        
        #embedded = [batch_size, seq len, emb dim]
        
        output, (_hidden, _cell) = self.lstm(embedded)
        
        #outputs = [batch size,seq len, hidden_dim * n directions]
        
        return (output[ : , : , : self.hidden_dim] 
                + output[ : , : , self.hidden_dim:]) # sum merge mode for BiLSTM, we can use: avg / mul / etc.

    




class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, output_dim, hidden_dim, n_layers = 2, dropout = 0):
        super().__init__()
        
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout = dropout, bidirectional = True)
        
        self.fc_out = nn.Linear(hidden_dim * self.seq_len, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        
        #input = [batch size, seq len, embed_dim]
        
        # embedded = self.dropout(input)
        embedded = input

        output, (_hidden, _cell) = self.lstm(embedded)
        
        #output = [batch size, seq len, hid dim * n directions]
        
        output = (output[:, :, : self.hidden_dim] + output[:, :, self.hidden_dim: ])
        
        # print("output", output.view(-1, self.seq_len * self. hidden_dim).shape)

        prediction = self.fc_out(output.view(-1, self.seq_len * self.hidden_dim))
        
        #prediction = [batch size, output dim]
        
        return prediction


class Encoder2Layer(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, hidden_dim, n_layers = 2, dropout = 0):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # self.embedding  = nn.Linear(input_dim, emb_dim)
        
        self.lstm1 = nn.LSTM(input_dim, 200, 1, dropout = dropout, bidirectional = False, batch_first = True)
        self.lstm2 = nn.LSTM(200, 100, 1, dropout = dropout, bidirectional = False, batch_first = True)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, seq):
        
        # seq : [batch size , seq len, input_dim]
        embedded = seq
        # embedded = self.embedding(seq)
        # embedded = self.dropout(embedded)
        
        #embedded = [batch_size, seq len, emb dim]
        
        output, (_hidden, _cell) = self.lstm1(embedded)
        output, (_hidden, _cell) = self.lstm2(output)
        
        #outputs = [batch size,seq len, hidden_dim * n directions]
        
        return output

    

class Decoder2Layer(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, output_dim, hidden_dim, n_layers = 2, dropout = 0):
        super().__init__()
        
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.lstm1 = nn.LSTM(input_dim, 200, 1, dropout = dropout, bidirectional = False, batch_first = True)
        self.lstm2 = nn.LSTM(200, 100, 1, dropout = dropout, bidirectional = False, batch_first = True)
        
        self.fc_out = nn.Linear(100, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        
        #input = [batch size, seq len, embed_dim]
        
        # embedded = self.dropout(input)
        embedded = input

        output, (_hidden, _cell) = self.lstm1(embedded)

        output, (_hidden, _cell) = self.lstm2(output)
        
        #output = [batch size, seq len, hid dim * n directions]
        
        # output = (output[:, :, : 2 * self.hidden_dim] + output[:, :, 2 * self.hidden_dim: ])
        
        # print("output", output.view(-1, self.seq_len * self. hidden_dim).shape)
        # print("output.size", output.size())
        # print(self.fc_out)

        # prediction = self.fc_out(output.contiguous().view(-1, self.seq_len * 100))
        prediction = self.fc_out(output.contiguous().view(-1, self.seq_len, 100))[:,-1,:]
        
        #prediction = [batch size, output dim]
        
        return prediction

class DeviceModel(nn.Module):

    def __init__(self,
                    seq_len, 
                    input_dim, 
                    emb_dim, 
                    hidden_dim,
                    output_dim,
                    att_out_dim = 64, 
                encoder_layer_n = 2, 
                decoder_layer_n = 2, 
                encoder_dropout = 0,
                decoder_dropout = 0,
                attention_head = 3,
                multihead_output_nodes = 32
                ):

        super().__init__()
        
        
        # encoder layer, default 2 layer BiLSTM
        self.encoder = Encoder2Layer(seq_len, input_dim, emb_dim, hidden_dim, 
                                n_layers = encoder_layer_n, 
                                dropout = encoder_dropout)
        # decoder layer, default 2 layer BiLSTM
        self.decoder = Decoder2Layer(seq_len, multihead_output_nodes, emb_dim, output_dim, hidden_dim, 
                                n_layers = decoder_layer_n, 
                                dropout = decoder_dropout)
        # self attention
        # self.attention = ScaledDotProductAttention(d_model = hidden_dim, 
        #                                             d_k = hidden_dim, 
        #                                             d_v = hidden_dim, 
        #                                             out_dim = att_out_dim,
        #                                             h = attention_head)
        self.attention = ScaledDotProductAttention(d_model = 100, 
                                                    d_k = 64, 
                                                    d_v = 64, 
                                                    out_dim = multihead_output_nodes,
                                                    h = attention_head)
        # self.attention = nn.MultiheadAttention( att_out_dim * attention_head, attention_head, batch_first = True)
        # self.fc_before_att = nn.Linear(100, att_out_dim * attention_head)
        # self.fc_after_att = nn.Linear(att_out_dim * attention_head, multihead_output_nodes)

    def forward(self, input_seq):
        
        out = self.encoder(input_seq)
        # out = self.fc_before_att(out)
        # out, _att_weight = self.attention(out, out, out)
        # out = self.fc_after_att(out)
        #
        out = self.attention(out, out, out)
        out = self.decoder(out)
        
        # out = self.decoder(input_seq)
        
        return out



class DeviceModelBiLSTM(nn.Module):

    def __init__(self,
                    seq_len, 
                    input_dim, 
                    emb_dim, 
                    hidden_dim,
                    output_dim,
                    att_out_dim = 64, 
                encoder_layer_n = 2, 
                decoder_layer_n = 2, 
                encoder_dropout = 0,
                decoder_dropout = 0,
                attention_head = 3,
                multihead_output_nodes = 32
                ):

        super().__init__()
        
        
        # encoder layer, default 2 layer BiLSTM
        self.encoder = Encoder(seq_len, input_dim, emb_dim, hidden_dim, 
                                n_layers = encoder_layer_n, 
                                dropout = encoder_dropout)
        # decoder layer, default 2 layer BiLSTM
        self.decoder = Decoder(seq_len, multihead_output_nodes, emb_dim, output_dim, hidden_dim, 
                                n_layers = decoder_layer_n, 
                                dropout = decoder_dropout)
        # self attention
        # self.attention = ScaledDotProductAttention(d_model = hidden_dim, 
        #                                             d_k = hidden_dim, 
        #                                             d_v = hidden_dim, 
        #                                             out_dim = att_out_dim,
        #                                             h = attention_head)
        self.attention = ScaledDotProductAttention(d_model = 100, 
                                                    d_k = 64, 
                                                    d_v = 64, 
                                                    out_dim = multihead_output_nodes,
                                                    h = attention_head)
        # self.attention = nn.MultiheadAttention( att_out_dim * attention_head, attention_head, batch_first = True)
        # self.fc_before_att = nn.Linear(100, att_out_dim * attention_head)
        # self.fc_after_att = nn.Linear(att_out_dim * attention_head, multihead_output_nodes)

    def forward(self, input_seq):
        
        out = self.encoder(input_seq)
        # out = self.fc_before_att(out)
        # out, _att_weight = self.attention(out, out, out)
        # out = self.fc_after_att(out)
        #
        out = self.attention(out, out, out)
        out = self.decoder(out)
        
        # out = self.decoder(input_seq)
        
        return out



