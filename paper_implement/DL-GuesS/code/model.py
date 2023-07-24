#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:40:02 2023

@author: welcome870117
"""

import torch
import torch.nn as nn


class single_price_model(nn.Module):
    def __init__(self, input_dim, batch_size, features):
        super(single_price_model, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.features = features
        
        # model architecture
        # nn.LSTM(input_size , hidden_size , num_layer)
        self.lstm = nn.LSTM(self.features, 100, 1, batch_first=True)
        self.gru = nn.GRU(100, 100, 1, batch_first=True)
        self.fc_100 = nn.Linear(100, 100)
        self.fc_1 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        
    def forward(self, price_data):
        # LSTM init parameter
        #h_lstm = torch.zeros(1, price_data.size(0), 100).requires_grad_()
        #c_lstm = torch.zeros(1, price_data.size(0), 100).requires_grad_()
        # LSTM 
        #lstm_out, (h_lstm, c_lstm) = self.lstm(price_data, (h_lstm.detach(), c_lstm.detach())) 
        lstm_out, (h_lstm, c_lstm) = self.lstm(price_data, None) 
        # gru init parameter
        h_gru = torch.zeros(1, lstm_out.size(0), 100).requires_grad_()
        gru_out , h_gru = self.gru(lstm_out, h_gru)
        #fc_100_out = self.fc_100(gru_out)
        gru_out = self.relu(gru_out)
        #fc_100_out = self.fc_100(lstm_out)
        output = self.fc_1(gru_out)
        #output = self.fc_1(gru_out)
        return output
        
    
class GuesS(nn.Module):

    def __init__(self,input_dim ,batch_size, features):
        super(GuesS , self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.features = features
        
        # layer
        # nn.LSTM(input_size , hidden_size , num_layer)
        self.lstm_BTC = nn.LSTM(self.features , 100 , 1 , batch_first=True)
        self.gru_BTC = nn.GRU(100 , 100 , 1 ,batch_first=True)
        self.fc_BTC = nn.Linear(100,100)
        self.dense5_BTC = nn.Linear(100*3,5)
        self.dense10_BTC = nn.Linear(5,10)
        
        
        self.lstm_AXS = nn.LSTM(self.features , 100 , 1 , batch_first=True)
        self.gru_AXS = nn.GRU(100 , 100 , 1 ,batch_first=True)
        self.fc_AXS = nn.Linear(100,100)
        self.dense5_AXS = nn.Linear(100*3,5)
        self.dense10_AXS = nn.Linear(5,10)
        
        #self.dense10_c
        self.dense10_c_BTC = nn.Linear(110,10) # 暫時改1
        self.dense10_c_AXS = nn.Linear(110,10)
        
        self.dense_final = nn.Linear(20,1)
        

    def forward(self,BTC_price,BTC_sentiment,AXS_price,AXS_sentiment):
        #print('BTC_sentiment_shape:',BTC_sentiment.shape)
        #print('BTC_price:',BTC_price.shape)
        # lstm 初始化參數
        #h_lstm_0 = torch.zeros(self.num_layers, BTC_price.size(0), self.hidden_dim).requires_grad_()
        #c_lstm_0 = torch.zeros(self.num_layers, BTC_price.size(0), self.hidden_dim).requires_grad_()
        h_lstm_0 = torch.zeros(1, BTC_price.size(0), 100).requires_grad_()
        c_lstm_0 = torch.zeros(1, BTC_price.size(0), 100).requires_grad_()
        
        BTC_lstm_out, (h_lstm_n, c_lstm_n) = self.lstm_BTC(BTC_price, (h_lstm_0.detach(), c_lstm_0.detach())) 
        AXS_lstm_out, (h_lstm_n, c_lstm_n) = self.lstm_AXS(BTC_price, (h_lstm_0.detach(), c_lstm_0.detach()))
        
        # gru 初始化參數
        h_gru0 = torch.zeros(1, BTC_lstm_out.size(0), 100).requires_grad_()
        
        BTC_gru_out , BTC_hn = self.gru_BTC(BTC_lstm_out,h_gru0)
        AXS_gru_out , AXS_hn = self.gru_AXS(AXS_lstm_out,h_gru0)
        
        block_BTC_out = self.fc_BTC(BTC_gru_out)
        block_AXS_out = self.fc_AXS(AXS_gru_out)
        
        # sentiment score 
        BTC_flatten_input = torch.flatten(BTC_sentiment,start_dim=2 , end_dim=-1)
        AXS_flatten_input = torch.flatten(AXS_sentiment,start_dim=2 , end_dim=-1)
        
        #print('BTC_flatten_input_shape:',BTC_flatten_input.shape)
        
        BTC_sentiemnt_out = self.dense5_BTC(BTC_flatten_input)
        BTC_sentiemnt_out = self.dense10_BTC(BTC_sentiemnt_out)
        
        
        AXS_sentiemnt_out = self.dense5_AXS(AXS_flatten_input)
        AXS_sentiemnt_out = self.dense10_AXS(AXS_sentiemnt_out)
        
        #BTC_sentiemnt_out = torch.reshape(BTC_sentiemnt_out,(1,1,10))
        #AXS_sentiemnt_out = torch.reshape(AXS_sentiemnt_out,(1,1,10))
        
        #print('block_BTC_out_shape:',block_BTC_out.shape)
        #print('BTC_sentiemnt_out_shape:',BTC_sentiemnt_out.shape)
        
        
        BTC_output_c1 = torch.cat([block_BTC_out,BTC_sentiemnt_out],2)
        BTC_outputs = self.dense10_c_BTC(BTC_output_c1)
           
        AXS_output_c1 = torch.cat([block_AXS_out,AXS_sentiemnt_out],2)
        AXS_outputs = self.dense10_c_AXS(AXS_output_c1)
        
        final_input = torch.cat([BTC_outputs,AXS_outputs],2)
        final_output = self.dense_final(final_input)
        
        
        return final_output