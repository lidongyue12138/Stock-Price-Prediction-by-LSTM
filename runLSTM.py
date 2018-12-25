from DataLoader import DataLoader
from LSTM_Model import Model
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt

"""
setting configs
"""
trainDataFileName = "train_data.csv"
testDataFileName = "test_data.csv"
modelSaveDir = "saved_models"
cols=["MidPrice", 
      "LastPrice",
      "AskVolume1",
      "BidVolume1",
      "AskPrice1",
      "BidPrice1",
      "Volume"]
sequenceLength = 30
batchSize = 32
epochNum = 2

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def main():
    if not os.path.exists(modelSaveDir): os.makedirs(modelSaveDir)

    data = DataLoader(
        trainDataFileName, 
        testDataFileName,
        cols
    )

    model = Model()
    model.build_model()
    # model.load_model()

    # x, y = data.get_train_data(
    #     seq_len=sequenceLength,
    #     normalise=True
    # )

    # print(x.shape)
    # print(y.shape)
    # print(y[-1])

    # out-of memory generative training
    print("...Begin Training...")
    steps_per_epoch = math.ceil((data.len_train - sequenceLength) / batchSize)
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=sequenceLength,
            batch_size=batchSize,
            normalise=True
        ),
        epochs=epochNum,
        batch_size=batchSize,
        steps_per_epoch=steps_per_epoch,
        save_dir=modelSaveDir
    )

    model.save_model()

    x_test, y_test = data.get_test_data(
        seq_len=10,
        normalise=True
    )

    # print(x_test.shape)

    # predictions = model.predict_sequence_full(x_test, sequenceLength)
    predictions = model.predict_data(x_test)
    np.savetxt("predictions.txt", predictions)


if __name__ == "__main__":
    main()