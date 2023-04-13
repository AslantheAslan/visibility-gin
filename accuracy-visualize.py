

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pickle

import os

directory = os.getcwd()


gin_directory = directory + "\output\GIN_accuracy.txt"
mlstm_fcn_directory = directory + "\output\MLSTM_FCN_accuracy.txt"
inception_directory = directory + "\output\InceptionTime_accuracy.txt"
lstm_fcn_directory = directory + "\output\LSTM_FCN_accuracy.txt"
fcn_directory = directory + "\output\FCN_accuracy.txt"
lstm_directory = directory + "\output\LSTM_accuracy.txt"
tcn_directory = directory + "\output\TCN_accuracy.txt"
mlp_directory = directory + "\output\MLP_accuracy.txt"
mWDN_directory = directory + "\output\mWDN_accuracy.txt"
XCM_directory = directory + "\output\XCM_accuracy.txt"


with open(gin_directory, "rb") as fp:   # Unpickling
  acc_array_gin = pickle.load(fp)

with open(mlstm_fcn_directory, "rb") as fp:   # Unpickling
  acc_array_mlstm_fcn = pickle.load(fp)

with open(inception_directory, "rb") as fp:   # Unpickling
  acc_array_inception = pickle.load(fp)

with open(lstm_directory, "rb") as fp:   # Unpickling
  acc_array_lstm = pickle.load(fp)

with open(tcn_directory, "rb") as fp:   # Unpickling
  acc_array_tcn = pickle.load(fp)

with open(mWDN_directory, "rb") as fp:   # Unpickling
  acc_array_mWDN = pickle.load(fp)

with open(XCM_directory, "rb") as fp:   # Unpickling
  acc_array_XCM = pickle.load(fp)

with open(fcn_directory, "rb") as fp:   # Unpickling
  acc_array_fcn = pickle.load(fp)

with open(mlp_directory, "rb") as fp:   # Unpickling
  acc_array_mlp = pickle.load(fp)

with open(lstm_fcn_directory, "rb") as fp:   # Unpickling
  acc_array_lstm_fcn = pickle.load(fp)


acc_arr_inception = np.array(acc_array_inception)[:,2]
acc_arr_mlstm_fcn = np.array(acc_array_mlstm_fcn)[:,2]
acc_arr_gin = np.array(acc_array_gin)[:,2]
acc_arr_lstm = np.array(acc_array_lstm)[:,2]
acc_arr_lstm_fcn = np.array(acc_array_lstm_fcn)[:,2]
acc_arr_mlp = np.array(acc_array_mlp)[:,2]
acc_arr_XCM = np.array(acc_array_XCM)[:,2]
acc_arr_mWDN = np.array(acc_array_mWDN)[:,2]
acc_arr_fcn = np.array(acc_array_fcn)[:,2]
acc_arr_tcn = np.array(acc_array_tcn)[:,2]

print(acc_arr_inception.shape, acc_arr_fcn.shape, acc_arr_tcn.shape, acc_arr_lstm.shape, acc_arr_XCM.shape, acc_arr_mWDN.shape, acc_arr_mlp.shape, acc_arr_gin.shape)

accuracies = np.stack((acc_arr_gin, acc_arr_inception, acc_arr_mlstm_fcn, acc_arr_lstm, acc_arr_lstm_fcn, acc_arr_mlp, acc_arr_XCM, acc_arr_mWDN, acc_arr_fcn, acc_arr_tcn), axis=1)

df = pd.DataFrame(accuracies, columns = ['VisGIN', 'InceptionTime', 'MLSTM-FCN', 'LSTM', 'LSTM-FCN', 'MLP', 'XCM', 'mWDN', 'FCN', 'TCN'])

columns = ['VisGIN', 'InceptionTime', 'MLSTM-FCN', 'LSTM', 'LSTM-FCN', 'MLP', 'XCM', 'mWDN', 'FCN', 'TCN']
df[columns] = df[columns].apply(lambda x: x*100)


print(df)
print(type(df))



_, ax = plt.subplots(figsize=(8,6))




figure = sns.kdeplot(ax=ax, data=df)
figure.set(xlim=(95, 100))
figure.set_ylabel("Density",fontsize=14)

sns.move_legend(ax, "center left", fontsize='x-large', title_fontsize='20')

plt.xlabel('Accuracy (%)', fontsize=14)
plt.savefig('3pulse_acc_denst.jpg', dpi=500)
plt.show()
