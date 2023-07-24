#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:57:09 2023

@author: welcome870117
"""

import pandas as pd

def split_different_coin(df):
    diff_symbol_data = {}
    start_index = 0
    for i in range(1, len(df)):
        if df['symbol'][i-1]!=df['symbol'][i]:
            diff_symbol_data[df['symbol'][i-1]] = df.iloc[start_index:i].reset_index(drop=True)
            start_index = i
    diff_symbol_data[df['symbol'].iloc[-1]] = df.iloc[start_index:].reset_index(drop=True)
    return diff_symbol_data

if __name__ == '__main__':
    # read dataset
    price_data = pd.read_csv('/Users/welcome870117/Desktop/git_project/Quantitative_trading_strategy/backtesting/crypto_total_hist_data_1200.csv', index_col=False)
    # split dataset in different coin
    dataset = split_different_coin(price_data)
    # required cryptocurrency
    required_crypto = ['BTC', 'DASH', 'LTC', 'BCH']
    # traverse dataset and get data
    required_dataset = dict()
    for k, v in dataset.items():
        if k in required_crypto:
            required_dataset[k] = v
    # build dataframe
    BTC_data = pd.DataFrame(required_dataset['BTC'])
    DASH_data = pd.DataFrame(required_dataset['DASH'])
    LTC_data = pd.DataFrame(required_dataset['LTC'])
    BCH_data = pd.DataFrame(required_dataset['BCH'])
    # save csv 
    BCH_data.to_csv('/Users/welcome870117/Desktop/git_project/finance-project/paper_implement/DL-GuesS/dataset/BCH_dataset.csv', index=False)
