import statistics

import numpy as np
import pickle

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import os
from utils import time_series_to_graphs
import networkx as nx
from torch_geometric.utils.convert import to_networkx


directory = os.getcwd()
read_directory = directory + "\data\ECG\corrected_baseline_pulses.txt"

with open(read_directory, "rb") as fp:   # Unpickling
    pulse_list = pickle.load(fp)

graphs = time_series_to_graphs(pulse_list)
print("type: ", type(graphs[0]), "length of graphs: ",len(graphs))

options = {"edgecolors": "tab:gray", "font_size": 8, "node_size": 100, "alpha": 0.9}


fig, axes = plt.subplots(5, 5, figsize=(16, 12), squeeze=True)
"""
1st individual's pulse graphs
"""

plt.subplot(551)
g1 = to_networkx(graphs[0], to_undirected=True)
nx.draw(g1, with_labels = True, node_color="tab:blue", edge_color="tab:blue", font_color="whitesmoke", **options)

plt.subplot(552)
g2 = to_networkx(graphs[2], to_undirected=True)
nx.draw(g2, with_labels = True, node_color="tab:blue", edge_color="tab:blue", font_color="whitesmoke", **options)

plt.subplot(553)
g3 = to_networkx(graphs[4], to_undirected=True)
nx.draw(g3, with_labels = True, node_color="tab:blue", edge_color="tab:blue", font_color="whitesmoke", **options)

plt.subplot(554)
g4 = to_networkx(graphs[6], to_undirected=True)
nx.draw(g4, with_labels = True, node_color="tab:blue", edge_color="tab:blue", font_color="whitesmoke", **options)

plt.subplot(555)
g5 = to_networkx(graphs[8], to_undirected=True)
nx.draw(g5, with_labels = True, node_color="tab:blue", edge_color="tab:blue", font_color="whitesmoke", **options)

"""
2nd individual's pulse graphs
"""

plt.subplot(556)
g6 = to_networkx(graphs[302], to_undirected=True)
nx.draw(g6, with_labels = True, node_color="tab:red", edge_color="tab:red", font_color="whitesmoke", **options)


plt.subplot(557)
g7 = to_networkx(graphs[304], to_undirected=True)
nx.draw(g7, with_labels = True, node_color="tab:red", edge_color="tab:red", font_color="whitesmoke", **options)


plt.subplot(558)
g8 = to_networkx(graphs[306], to_undirected=True)
nx.draw(g8, with_labels = True, node_color="tab:red", edge_color="tab:red", font_color="whitesmoke", **options)


plt.subplot(559)
g9 = to_networkx(graphs[308], to_undirected=True)
nx.draw(g9, with_labels = True, node_color="tab:red", edge_color="tab:red", font_color="whitesmoke", **options)


plt.subplot(5, 5, 10)
g10 = to_networkx(graphs[310], to_undirected=True)
nx.draw(g10, with_labels = True, node_color="tab:red", edge_color="tab:red", font_color="whitesmoke", **options)


"""
3rd individual's pulse graphs
"""
plt.subplot(5, 5, 11)
g11 = to_networkx(graphs[122], to_undirected=True)
nx.draw(g11, with_labels = True, node_color="tab:green", edge_color="tab:green", font_color="whitesmoke", **options)

plt.subplot(5, 5, 12)
g12 = to_networkx(graphs[124], to_undirected=True)
nx.draw(g12, with_labels = True, node_color="tab:green", edge_color="tab:green", font_color="whitesmoke", **options)

plt.subplot(5, 5, 13)
g13 = to_networkx(graphs[126], to_undirected=True)
nx.draw(g13, with_labels = True, node_color="tab:green", edge_color="tab:green", font_color="whitesmoke", **options)

plt.subplot(5, 5, 14)
g14 = to_networkx(graphs[128], to_undirected=True)
nx.draw(g14, with_labels = True, node_color="tab:green", edge_color="tab:green", font_color="whitesmoke", **options)

plt.subplot(5, 5, 15)
g15 = to_networkx(graphs[130], to_undirected=True)
nx.draw(g15, with_labels = True, node_color="tab:green", edge_color="tab:green", font_color="whitesmoke", **options)


"""
4th individual's pulse graphs
"""

plt.subplot(5, 5, 16)
g16 = to_networkx(graphs[182], to_undirected=True)
nx.draw(g16, with_labels = True, node_color="tab:cyan", edge_color="tab:cyan", font_color="whitesmoke", **options)

plt.subplot(5, 5, 17)
g17 = to_networkx(graphs[184], to_undirected=True)
nx.draw(g17, with_labels = True, node_color="tab:cyan", edge_color="tab:cyan", font_color="whitesmoke", **options)

plt.subplot(5, 5, 18)
g18 = to_networkx(graphs[186], to_undirected=True)
nx.draw(g18, with_labels = True, node_color="tab:cyan", edge_color="tab:cyan", font_color="whitesmoke", **options)

plt.subplot(5, 5, 19)
g19 = to_networkx(graphs[188], to_undirected=True)
nx.draw(g19, with_labels = True, node_color="tab:cyan", edge_color="tab:cyan", font_color="whitesmoke", **options)

plt.subplot(5, 5, 20)
g20 = to_networkx(graphs[190], to_undirected=True)
nx.draw(g20, with_labels = True, node_color="tab:cyan", edge_color="tab:cyan", font_color="whitesmoke", **options)



"""
5th individual's pulse graphs
"""
plt.subplot(5, 5, 21)
g21 = to_networkx(graphs[242], to_undirected=True)
nx.draw(g21, with_labels = True, node_color="tab:pink", edge_color="tab:pink", font_color="whitesmoke", **options)

plt.subplot(5, 5, 22)
g22 = to_networkx(graphs[244], to_undirected=True)
nx.draw(g22, with_labels = True, node_color="tab:pink", edge_color="tab:pink", font_color="whitesmoke", **options)

plt.subplot(5, 5, 23)
g23 = to_networkx(graphs[246], to_undirected=True)
nx.draw(g23, with_labels = True, node_color="tab:pink", edge_color="tab:pink", font_color="whitesmoke", **options)

plt.subplot(5, 5, 24)
g24 = to_networkx(graphs[248], to_undirected=True)
nx.draw(g24, with_labels = True, node_color="tab:pink", edge_color="tab:pink", font_color="whitesmoke", **options)

plt.subplot(5, 5, 25)
g25 = to_networkx(graphs[250], to_undirected=True)
nx.draw(g25, with_labels = True, node_color="tab:pink", edge_color="tab:pink", font_color="whitesmoke", **options)


plt.savefig('visgraphs.jpg', dpi=500)
plt.show()

"""
ECG plots of corresponding patients

"""
ts_test_sample = np.arange(0, 30, 1)

fig2, axs2 = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(16, 12), squeeze=True)
print(len(pulse_list), len(pulse_list[0]))
axs2[0, 0].plot(ts_test_sample, pulse_list[0][0], 'tab:blue')

axs2[0, 1].plot(ts_test_sample, pulse_list[0][2], 'tab:blue')

axs2[0, 2].plot(ts_test_sample, pulse_list[0][4], 'tab:blue')

axs2[0, 3].plot(ts_test_sample, pulse_list[0][6], 'tab:blue')

axs2[0, 4].plot(ts_test_sample, pulse_list[0][8], 'tab:blue')

axs2[1, 0].plot(ts_test_sample, pulse_list[5][2], 'tab:red')

axs2[1, 1].plot(ts_test_sample, pulse_list[5][4], 'tab:red')

axs2[1, 2].plot(ts_test_sample, pulse_list[5][6], 'tab:red')

axs2[1, 3].plot(ts_test_sample, pulse_list[5][8], 'tab:red')

axs2[1, 4].plot(ts_test_sample, pulse_list[5][10], 'tab:red')

axs2[2, 0].plot(ts_test_sample, pulse_list[2][2], 'tab:green')

axs2[2, 1].plot(ts_test_sample, pulse_list[2][4], 'tab:green')

axs2[2, 2].plot(ts_test_sample, pulse_list[2][6], 'tab:green')

axs2[2, 3].plot(ts_test_sample, pulse_list[2][8], 'tab:green')

axs2[2, 4].plot(ts_test_sample, pulse_list[2][10], 'tab:green')

axs2[3, 0].plot(ts_test_sample, pulse_list[3][2], 'tab:cyan')

axs2[3, 1].plot(ts_test_sample, pulse_list[3][4], 'tab:cyan')

axs2[3, 2].plot(ts_test_sample, pulse_list[3][6], 'tab:cyan')

axs2[3, 3].plot(ts_test_sample, pulse_list[3][8], 'tab:cyan')

axs2[3, 4].plot(ts_test_sample, pulse_list[3][10], 'tab:cyan')

axs2[4, 0].plot(ts_test_sample, pulse_list[4][2], 'tab:pink')

axs2[4, 1].plot(ts_test_sample, pulse_list[4][4], 'tab:pink')

axs2[4, 2].plot(ts_test_sample, pulse_list[4][6], 'tab:pink')

axs2[4, 3].plot(ts_test_sample, pulse_list[4][8], 'tab:pink')

axs2[4, 4].plot(ts_test_sample, pulse_list[4][10], 'tab:pink')



# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs2.flat:
    ax.label_outer()




fig2.text(0.5, 0.04, 'ts (sample)', ha='center')
fig2.text(0.04, 0.5, 'Magnitude (mV)', va='center', rotation='vertical')

plt.savefig('ECGs.jpg', dpi=500)
plt.show()
