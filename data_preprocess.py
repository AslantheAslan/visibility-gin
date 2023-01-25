
"""

IMPORTANT NOTE:

As CU-ECG dataset is a private dataset, we're unable to provide the full details of the dataset. This script is the
data preprocessor in which we filter, downsample and get pulse segments. That being said, you can find the pulse
segments in "data" folder so that you can reproduce the outputs. Note that the original dataset takes 79 GB whereas
given data goes up to 45 MB at most.

More explanation regarding dataset is available in readme.txt.

"""

import numpy as np
import pandas as pd

from utils import similar_pulse_manhattan, calc_baseline

import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

import pywt

from sklearn.metrics.pairwise import manhattan_distances

import glob
from matplotlib import pyplot

import pickle

df = pd.read_csv("D:\ECGDATA\ECGDB\\1\\1.csv")
df.columns = ['Timestamp', 'Peak(mV)']
del df["Timestamp"]
df['Peak(mV)'].plot()


f2, f1 = scipy.signal.butter(3, 0.01)
filtered = scipy.signal.filtfilt(f2, f1, df['Peak(mV)'])
plt.show()

### Raw versus Filtered ###

times = np.arange(0, len(df['Peak(mV)']), 1)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(times, df['Peak(mV)'])
plt.title("ECG Signal with Noise")
plt.margins(0, .05)

plt.subplot(122)
plt.plot(times, filtered)
plt.title("Filtered ECG Signal")
plt.margins(0, .05)

plt.tight_layout()
plt.show()

### Filtered and Downsampled ECG Signal ###

filtered = filtered[:500000]

R = 1000
filtered_downsampled = filtered.reshape(-1, R).mean(axis=1)

plt.title("Downsampled ECG data")
ts = np.arange(0, len(filtered_downsampled), 1)
plt.plot(ts, filtered_downsampled, color="blue")

plt.show()

### baseline correction ###

baseline = calc_baseline(filtered_downsampled)

ecg_out = filtered_downsampled - baseline

plt.rcParams["figure.figsize"] = (20,3)
plt.subplot(2, 1, 1)
plt.plot(filtered_downsampled, "b-", label="signal")
plt.plot(baseline, "r-", label="baseline")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ecg_out, "b-", label="signal - baseline")
plt.legend()

plt.show()

### test sample ###

test_sample = ecg_out[50:80]

plt.title("Test sample")
ts_test_sample = np.arange(0, len(test_sample), 1)
plt.plot(ts_test_sample, test_sample, color="red")

plt.show()

R = 1000
test_list = [test_sample.tolist()]
glued_data = pd.DataFrame()
man_distances = []
indices = []
for folder_name in glob.glob("D:\ECGDATA\ECGDB\*"):
    print(folder_name)
    for file_name in glob.glob(folder_name + '/*.csv'):
        dfx = pd.read_csv(file_name)
        dfx.columns = ['Timestamp', 'Peak(mV)']
        del dfx["Timestamp"]

        dx_filtered = scipy.signal.filtfilt(f2, f1, dfx['Peak(mV)'])
        dx_times = np.arange(0, len(dfx['Peak(mV)']), 1)

        if len(dx_filtered)>500000:
            dx_filtered = dx_filtered[:500000]

        dx_filtered_aranged = np.append(dx_filtered,np.zeros(500000-len(dx_filtered)))
        dx_filtered_downsampled = dx_filtered_aranged.reshape(-1, R).mean(axis=1)

        baseline = calc_baseline(dx_filtered_downsampled)
        ecg_out = dx_filtered_downsampled - baseline

        indices += [similar_pulse_manhattan(ecg_out, test_list)[1]]
        man_distances += [similar_pulse_manhattan(ecg_out, test_list)[0]]

print("Length of man_distances", len(man_distances))

besides_p1 = man_distances
besides_p1 = np.delete(besides_p1, slice(60), 0)

print("Maximum similarity:", np.max(besides_p1))
print("Mean similarity:", np.mean(besides_p1))
print("Similarity variance:", np.var(besides_p1))
print("Minimum similarity:", np.min(besides_p1))

bins = np.linspace(0, 10, 100)

pyplot.figure(figsize=(8, 6), dpi=70)
pyplot.hist(besides_p1, bins, alpha=0.5, label='Impostor match')
pyplot.hist(man_distances[0:59], bins, alpha=0.5, label='Genuine match')
pyplot.legend(loc='upper right')
pyplot.show()

pulse_extract = []
j=0
for folder_name in glob.glob("D:\ECGDATA\ECGDB\*"):
    print(folder_name)
    for file_name in glob.glob(folder_name +'/*.csv'):

        dfx = pd.read_csv(file_name)
        dfx.columns = ['Timestamp', 'Peak(mV)']
        del dfx["Timestamp"]

        dx_filtered = scipy.signal.filtfilt(f2, f1, dfx['Peak(mV)'])
        dx_times = np.arange(0, len(dfx['Peak(mV)']), 1)

        if len(dx_filtered) > 500000:
            dx_filtered = dx_filtered[:500000]

        dx_filtered_aranged = np.append(dx_filtered, np.zeros(500000 - len(dx_filtered)))
        dx_filtered_downsampled = dx_filtered_aranged.reshape(-1, R).mean(axis=1)

        baseline = calc_baseline(dx_filtered_downsampled)
        ecg_out = dx_filtered_downsampled - baseline

        pulse_extract += [list(ecg_out[indices[j]*5:indices[j]*5+30])]
        j=j+1

### Pulse Extract plots

fig, axs = plt.subplots(2, 4)
axs[0, 0].plot(ts_test_sample, pulse_extract[0])
axs[0, 0].set_title('1st patient')
axs[0, 1].plot(ts_test_sample, pulse_extract[20], 'tab:blue')
axs[0, 1].set_title('1st patient')
axs[0, 2].plot(ts_test_sample, pulse_extract[40], 'tab:blue')
axs[0, 2].set_title('1st patient')
axs[0, 3].plot(ts_test_sample, pulse_extract[55], 'tab:blue')
axs[0, 3].set_title('1st patient')
axs[1, 0].plot(ts_test_sample, pulse_extract[725], 'tab:green')
axs[1, 0].set_title('Impostor')
axs[1, 1].plot(ts_test_sample, pulse_extract[785], 'tab:red')
axs[1, 1].set_title('2nd Impostor')
axs[1, 2].plot(ts_test_sample, pulse_extract[845], 'tab:orange')
axs[1, 2].set_title('3rd Impostor')
axs[1, 3].plot(ts_test_sample, pulse_extract[905], 'tab:purple')
axs[1, 3].set_title('4th Impostor')

for ax in axs.flat:
    ax.set(xlabel='ts (sample)', ylabel='Magnitude')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

"""
You can comment out this part if you want to save the preprocessed data.

with open("D:\ECGDATA\ECGPulses\pulse_extract.txt", "wb") as fp:   #Pickling
  pickle.dump(pulse_extract, fp)
"""