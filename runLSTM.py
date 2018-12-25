from NewDataLoader import DataLoader
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
    if not os.path.exists(modelSaveDir): 
        os.makedirs(modelSaveDir)

    data = DataLoader(
        trainDataFileName, 
        testDataFileName,
    )

    morning_model = Model("morning_model")
    afternoon_model = Model("afternoon_model")
    morning_model.load_model()
    afternoon_model.load_model()
    # morning_model.build_model()
    # afternoon_model.build_model()

    # # out-of memory generative training
    # print("...Begin Training Morning LSTM...")
    # for i in range(len(data.morning_data)):
    #     steps_per_epoch = data.morning_data[i][1] - data.morning_data[i][0] - 30
    #     morning_model.train_generator(
    #         data_gen = data.next_morning_batch(),
    #         epochs = 2,
    #         batch_size = 100,
    #         steps_per_epoch = steps_per_epoch,
    #         save_dir = modelSaveDir
    #     )

    # print("...Begin Training Afternoon LSTM...")
    # for i in range(len(data.afternoon_data)):
    #     steps_per_epoch = data.afternoon_data[i][1] - data.afternoon_data[i][0] - 30
    #     morning_model.train_generator(
    #         data_gen = data.next_afternoon_batch(),
    #         epochs = 2,
    #         batch_size = 100,
    #         steps_per_epoch = steps_per_epoch,
    #         save_dir = modelSaveDir
    #     )


    # morning_model.save_model()
    # afternoon_model.save_model()

    predictions = []
    for _ in range(int(len(data.test_date_time)/10)):
        tmp_test, if_noon = data.get_next_test_data()
        tmpPrediction = []
        if if_noon:
            tmpPrediction = morning_model.predict_data(tmp_test)
        else:
            tmpPrediction = afternoon_model.predict_data(tmp_test)
        print(tmpPrediction)
        print(if_noon)
        predictions += np.ndarray.tolist(tmpPrediction[0])
    predictions = np.array(predictions)
    np.savetxt("predictions.txt", predictions)


if __name__ == "__main__":
    main()