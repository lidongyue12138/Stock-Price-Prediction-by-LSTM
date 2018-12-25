#%%
from NewDataLoader import DataLoader
from LSTM_Model import Model
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt

trainDataFileName = "train_data.csv"
testDataFileName = "test_data.csv"
modelSaveDir = "saved_models"

if not os.path.exists(modelSaveDir): 
    os.makedirs(modelSaveDir)

data = DataLoader(
    trainDataFileName, 
    testDataFileName,
)

tmp_test, if_noon = data.get_next_test_data()
tmp_test = np.array(tmp_test)

train_datas, train_labels = data.get_morning_train_data()
print(tmp_test[-1, -1, :, 0])
print(train_datas[-1, :, 0])
print(train_labels[-1])
