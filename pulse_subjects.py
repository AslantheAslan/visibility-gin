
"""

IMPORTANT NOTE:

As CU-ECG dataset is a private dataset, we're unable to provide the full details of the dataset. This script is the
data preprocessor in which we filter, downsample and get pulse segments. That being said, you can find the pulse
segments in "data" folder so that you can reproduce the outputs. Note that the original dataset takes 79 GB whereas
given data goes up to 45 MB at most.

More explanation regarding dataset is available in readme.txt.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
with open("D:\ECGDATA\ECGPulses\pulse_extract.txt", "rb") as fp:   # Unpickling
  pulse_extract = pickle.load(fp)
"""

with open("D:\ECGDATA\ECGPulses\\three_pulse_extract.txt", "rb") as fp:   # Unpickling
  pulse_extract = pickle.load(fp)

print("multi_pulse_extract size:  ",len(pulse_extract))

ts_test_sample = np.arange(0, 30, 1)

fig, axs = plt.subplots(2, 4)
axs[0, 0].plot(ts_test_sample, pulse_extract[0])
axs[0, 0].set_title('1st subject')
axs[0, 1].plot(ts_test_sample, pulse_extract[360], 'tab:blue')
axs[0, 1].set_title('2nd subject')
axs[0, 2].plot(ts_test_sample, pulse_extract[540], 'tab:blue')
axs[0, 2].set_title('3rd subject')
axs[0, 3].plot(ts_test_sample, pulse_extract[720], 'tab:blue')
axs[0, 3].set_title('4th subject')
axs[1, 0].plot(ts_test_sample, pulse_extract[900], 'tab:green')
axs[1, 0].set_title('5th subject')
axs[1, 1].plot(ts_test_sample, pulse_extract[1080], 'tab:red')
axs[1, 1].set_title('6th subject')
axs[1, 2].plot(ts_test_sample, pulse_extract[1260], 'tab:orange')
axs[1, 2].set_title('7th subject')
axs[1, 3].plot(ts_test_sample, pulse_extract[1440], 'tab:purple')
axs[1, 3].set_title('8th subject')


for ax in axs.flat:
    ax.set(xlabel='ts (sample)', ylabel='Magnitude')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig.suptitle('Extracted Pulses')
plt.show()

patients = []

for j in range(0,18000,180):

  patients += [pulse_extract[j:j+180]]

print(len(patients), len(patients[0]), len(patients[0][0]))

"""
You can comment out this part if you want to save the extracted pulses. 
We already provided the extracted pulses in data folder.

with open("D:\ECGDATA\ECGPulses\multi_patients_3.txt", "wb") as fp:   #Pickling
  pickle.dump(patients, fp)
"""