#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:24:01 2023

@author: welcome870117
"""
import  math

def price_normalization(price_data):
    digits = int(math.log10(max(list(price_data['High']))))
    normalize_number = math.pow(10, digits)
    normalize_columns = price_data[['Open', 'High', 'Close', 'Low']]/normalize_number
    price_data[['Open', 'High', 'Close', 'Low']] = normalize_columns[['Open', 'High', 'Close', 'Low']]
    
    return price_data


def split_training_teseting_dataset(df, train_date, test_date):
    '''

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    train_date : TYPE
        DESCRIPTION.
    teset_date : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # get index
    for i in range(len(df)):
        if df['Date'][i] == train_date:
            train_start_index = i
            
        elif df['Date'][i] == test_date:
            test_start_index = i
    
    # split training / testing
    train_dataset = df.iloc[train_start_index:test_start_index].reset_index(drop=True)
    test_dataset = df.iloc[test_start_index:].reset_index(drop=True)
    
    return train_dataset, test_dataset


def split_dataset_xy(data, window_size, datatype):
    x = []
    y = []
    
    if datatype=='nparray':
        for i in range(len(data)-window_size):
            x.append(data[i:i+window_size,:])
            y.append(data[i+window_size,:])
        return x,y            
    
    elif datatype=='dataframe':
        for i in range(len(data)-window_size):
            x.append(data.iloc[i:i+window_size,:])
            y.append(data.iloc[i+window_size,:])
        return x,y    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    