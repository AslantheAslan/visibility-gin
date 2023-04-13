import statistics

import numpy as np
import pickle

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import os

directory = os.getcwd()
full_directory = directory + "\output\GIN_accuracy.txt"

with open(full_directory, "rb") as f:   #Pickling
  accuracy = pickle.load(f) # check if accuracies saved successfully

list = [[0] * 100 for _ in range(100)]

for i in range(len(accuracy)):
  list[accuracy[i][0]][accuracy[i][1]] = accuracy[i][2]
  list[accuracy[i][1]][accuracy[i][0]] = accuracy[i][2]

array = np.array(list)

ax = sns.heatmap(array,vmin=0.7, vmax=1 ,cmap="YlGnBu")
#plt.savefig('FCN_3.jpg', dpi=300)
plt.show()

total, tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0, 0

for j in range(len(accuracy)):

  total += accuracy[j][2]
  tp_total += accuracy[j][3]
  fp_total += accuracy[j][4]
  fn_total += accuracy[j][5]
  tn_total += accuracy[j][6]

average = total/len(accuracy)

fnmr = fn_total/(fn_total+tp_total)
fmr  = fp_total/(fp_total+tn_total)

tpr = tp_total/(tp_total+fn_total)
fpr = fp_total/(fp_total+tn_total)
auc_roc = (1+tpr-fpr)/2

dev = []
for i in range(len(accuracy)):
  dev += [accuracy[i][2]]

print(dev)
print(len(dev))
print("Standard Deviation of Accuracies:  ", statistics.stdev(dev))

print("Average accuracy:   ", average,"   Average FNMR:   ", fnmr, "   Average FMR:   ", fmr)
print("Average tpr:   ", tpr,"   Average fpr:   ", fpr, "   Average auc_roc:   ", auc_roc)