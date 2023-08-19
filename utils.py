
import numpy as np

"""
import scipy.io.wavfile
from scipy import signal

import matplotlib.pyplot as plt
"""

import pywt

from sklearn.metrics.pairwise import manhattan_distances

from torch_geometric.utils.convert import from_networkx
from ts2vg import NaturalVG, HorizontalVG

import networkx as nx

"""
from scipy.interpolate import interp1d
import peakutils
"""

# Function for binary_search
def binary_search(arr, low, high, ele):
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == ele:
            return mid
        elif arr[mid] > ele:
            high = mid
        else:
            low = mid + 1
    return -1


# Function to get smallest n numbers
def getSmallest(arr, asize, n):
    mins = []

    # Make copy of array
    copy_arr = arr.copy()

    # Sort copy array
    copy_arr.sort()

    # For each arr[i] find whether
    # it is a part of n-smallest
    # with binary search
    for i in range(asize):
        if binary_search(copy_arr, low=0,
                         high=n, ele=arr[i]) > -1:
            # print(arr[i], end = " ")
            mins += [arr[i]]

    # get the original indices of sorted elements
    org_indices = sorted(range(asize), key=lambda k: arr[k])
    return mins, org_indices


def multi_similar_pulse_manhattan(filtered_downsampled, test_list):
    arr_init_res = []
    sliding_size = 5
    best_res = 100
    i = 0
    for i in range(94):
        filtered_downsampled_list = [filtered_downsampled[(sliding_size * i):(30 + sliding_size * i)].tolist()]
        init_res = manhattan_distances(filtered_downsampled_list, test_list)[0][0]

        arr_init_res += [init_res]

    return arr_init_res

def similar_pulse_manhattan(filtered_downsampled, test_list):
  sliding_size = 5
  best_res = 100
  i=0
  for i in range(94):
    filtered_downsampled_list = [filtered_downsampled[(sliding_size*i):(30+sliding_size*i)].tolist()]
    init_res = manhattan_distances(filtered_downsampled_list, test_list)[0][0]
    if (init_res < best_res):
      best_res = init_res
      last = i
  #print(best_res,i)
  return best_res, last

def calc_baseline(signal):
    """
    Calculate the baseline of signal.
    Args:
        signal (numpy 1d array): signal whose baseline should be calculated
    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

def time_series_to_graphs(pulse_list):
    graphs = []
    for subject in range(len(pulse_list)):
        for pulse in range(len(pulse_list[0])):
            g = NaturalVG()
            g.build(pulse_list[subject][pulse])
            nx_g = g.as_networkx()
            #high = np.max(pulse_list[subject][pulse])
            for i in range(nx_g.number_of_nodes()):
                if i < 29 :
                    slope1 = pulse_list[subject][pulse][i+1] - pulse_list[subject][pulse][i]
                if i < 28 :
                    slope2 = (pulse_list[subject][pulse][i+2] - pulse_list[subject][pulse][i])/2
                if i < 27 :
                    slope3 = pulse_list[subject][pulse][i+3] - pulse_list[subject][pulse][i]/3

                nx_g.add_node(i, x=[pulse_list[subject][pulse][i], slope1, slope2, slope3])

            H = nx.Graph()
            H.add_nodes_from(sorted(nx_g.nodes(data=True)))
            H.add_edges_from(nx_g.edges(data=True))

            H.nodes()

            pyg_graph = from_networkx(H)
            graphs += [pyg_graph]

    return graphs

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(4, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GAT_net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT_net, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(4, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GIN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(4, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 3)
        self.lin2 = Linear(hidden_channels * 3, 2)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)


        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
