#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 23:10:10 2023

@author: welcome870117
"""

# import python package
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# import python file
from data_preprocessing import price_normalization, split_training_teseting_dataset, split_dataset_xy
from model import single_price_model 


class single_model_dataset(Dataset):
    def __init__(self, features, label):
        self.features = features
        self.label = label
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 從 data DataFrame 中提取特徵，並轉換為 PyTorch Tensor
        features = torch.tensor(self.features[idx].iloc[:, [2, 3, 4, 5]].values, dtype=torch.float32)
        
        # 從 label DataFrame 中提取標籤，並轉換為 PyTorch Tensor
        label = torch.tensor(self.label[idx]['Close'], dtype=torch.float32)  # 假設標籤為整數型態
        return features, label


if __name__ == '__main__':
    # read data
    DASH_price = pd.read_csv('/Users/welcome870117/Desktop/git_project/finance-project/paper_implement/DL-GuesS/dataset/DASH_dataset.csv', index_col = False)
    # data normalization 
    DASH_price_n = price_normalization(DASH_price)
    # split trainset, testset
    training_dataset, testing_dataset = split_training_teseting_dataset(DASH_price_n, '2014-02-14', '2023-04-01')
    # get dataset x, y
    train_x, train_y = split_dataset_xy(training_dataset, 1, 'dataframe')
    test_x, test_y = split_dataset_xy(testing_dataset, 1, 'dataframe')
    # torch dataset
    dataset_single = single_model_dataset(train_x, train_y)
    dataset_single_test = single_model_dataset(test_x, test_y)
    # torch dataloader
    dataset_single_dataloader = DataLoader(dataset_single,  batch_size=16, shuffle=False)
    dataset_single_test_dataloader = DataLoader(dataset_single_test,  batch_size=1, shuffle=False)
    
    '''
    for batch_features, batch_labels in dataset_single_dataloader:
        print("Batch Features:")
        print(batch_features)
        print("Batch Labels:")
        print(batch_labels)
        print("---------------")
    '''
    
    # 
    model = single_price_model(3,16,4)
    # loss
    criterion = torch.nn.MSELoss(reduction='mean')
    #  optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=0.00005)
    
    num_epochs = 50
    
    for t in range(num_epochs):
        total_loss = []
        sum_loss = 0
        for i,data in enumerate(dataset_single_dataloader):
            feature, label = data           
            model_predict = model(feature)
            loss = criterion(model_predict,label)
            sum_loss+=loss
            total_loss.append(sum_loss)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step() 
        print('total_loss:',sum_loss)
    
    model_predict = []
    real_price = []
    for i,data in enumerate(dataset_single_test_dataloader):
        features, label = data
        predict = model(features)
        model_predict.append(predict.detach().numpy().reshape(-1,)[0])  
        real_price.append(label.detach().numpy().reshape(-1,)[0])
        
    
    plt.plot(real_price, c='black')
    plt.plot(model_predict, c='blue')








