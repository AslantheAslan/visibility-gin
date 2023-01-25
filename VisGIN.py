
import numpy as np
import pandas as pd
import pickle

import os
import torch

import random

from torch_geometric.loader import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from utils import time_series_to_graphs, GCN, GraphSAGE, GAT_net, GIN

"""
To try other graph ML models, you can refer to some of the architectures in utils.py.
"""

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.

        data = data.to(dev)
        out, _ = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
        #print(torch.Tensor(data.y).shape)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    preds_lst = []
    correct = 0
    CM = 0
    for data in loader:  # Iterate in batches over the training/test dataset.

        data = data.to(dev)
        out, _ = model(data.x.float(), data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        preds_lst += [pred]
        CM += confusion_matrix(data.y.cpu(), pred.cpu(), labels=[0, 1])

        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    return correct / len(loader.dataset), pred, preds_lst, CM  # Derive ratio of correct predictions.


os.environ['TORCH'] = torch.__version__
print(torch.__version__)
print("CUDA available? ", torch.cuda.is_available())
#print("Device name: ", torch.cuda.get_device_name(0))

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device in use is ", dev)

directory = os.getcwd()
read_directory = directory + "\data\ECG\multi_patients_3.txt"
save_directory = directory + "\output"

with open(read_directory, "rb") as fp:   # Unpickling
    pulse_list = pickle.load(fp)

graphs = time_series_to_graphs(pulse_list)

print('============================')
print("Length of graphs list:  ", len(graphs), "first graph in the list:  ", graphs[0], "features of the first graph:  ",graphs[0].x)
print('============================')

accuracy_tuples = []

for genuine in range(100):
    for impostor in range(genuine+1,100):

        match_set = graphs[genuine*60: genuine*60+60] + graphs[impostor*60: impostor*60+60]
        print("Length of match set: ", len(match_set))

        for i in range(60):
            match_set[i].y = 0
        for j in range(60,120):
            match_set[j].y = 1

        random.shuffle(match_set)

        train_dataset = match_set[:96]
        test_dataset = match_set[96:120]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


        model = GIN(hidden_channels=32)
        model.to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        best = 0
        for epoch in range(1, 501):
            train()
            train_acc, _, _, _ = test(train_loader)
            test_acc, _, preds_test, CM = test(test_loader)
            if test_acc > best:
                best = test_acc
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Acc: {best:04f}')

            if best == 1.00:
                break

        tn = CM[1][1]
        tp = CM[0][0]
        fp = CM[0][1]
        fn = CM[1][0]

        print("Genuine patient number:", genuine, "impostor patient number:", impostor)
        print("Test accuracy:", best)

        print("False Positive: ", fp, "  False Negative: ", fn, "  True Positive: ", tp,
              "  True Negative: ", tn)

        accuracy_tuples += [(genuine, impostor, best, tp, fp, fn, tn)]

print(accuracy_tuples)

with open(save_directory + "\\" + "GIN"+ "_accuracy.txt", "wb") as fp:  # Pickling
    pickle.dump(accuracy_tuples, fp)  # save the accuracy scores per match
