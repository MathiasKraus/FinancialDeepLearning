# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 08:58:39 2016

@author: mkraus
"""

from __future__ import division
from __future__ import print_function

import pandas as pd
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

validation_data_path = 'val_data.csv'
model_path = './models/AbnormalClassificationTL.hdf5'
target = 'AbnormalPositive'
max_len = 200

if __name__ == "__main__":
    df = pd.read_csv(validation_data_path)

    tk = pickle.load(open('tokenizer.pickle', 'rb'))
    
    X = tk.texts_to_sequences(df['Text'])   
    X = pad_sequences(X, maxlen=max_len)

    Y = df[target].apply(lambda x: 1 if x == True else 0)
    
    model = load_model(model_path)
    score = model.evaluate(X, Y)
    print(score[1])
