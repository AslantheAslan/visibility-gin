
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

from utils import *

import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import pickle

import pywt

from sklearn.metrics.pairwise import manhattan_distances

import glob
from matplotlib import pyplot

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

test_sample = filtered_downsampled[50:80]

plt.title("Test sample")
ts_test_sample = np.arange(0, len(test_sample), 1)
plt.plot(ts_test_sample, test_sample, color="red")

plt.show()

R = 1000
test_list = [test_sample.tolist()]
glued_data = pd.DataFrame()

diff_score = []
org_ind = []

n = 3

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

        dx_scores = multi_similar_pulse_manhattan(dx_filtered_downsampled, test_list)
        diff_score += [getSmallest(dx_scores, len(dx_scores), n)[0]]
        org_ind += [getSmallest(dx_scores, len(dx_scores), n)[1][:n]]

# using org_ind array, lets create the pulses as follows:

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

        for i in range(n):
            pulse_extract += [list(dx_filtered_downsampled[org_ind[j][i] * 5:org_ind[j][i] * 5 + 30])]
        j = j + 1

plt.title("Extracted Pulse")
ts_test_sample = np.arange(0, len(pulse_extract[200]), 1)
plt.plot(ts_test_sample, pulse_extract[200], color="red")

plt.show()

"""
with open("D:\ECGDATA\ECGPulses\\three_pulse_extract.txt", "wb") as fp:   #Pickling
  pickle.dump(pulse_extract, fp)
"""