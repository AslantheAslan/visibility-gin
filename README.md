# Visibility-GIN
This repository includes the Python implementation of a biometric authentication scheme, (VisGIN:  Visibility Graph Neural Network on One-dimensional Data for Biometric Authentication), using the proposed graph ML method.

## Training baseline models and VisGIN

The following can be run to check baseline models:

```
python tsai_trial.py
```

If you want to reproduce results for different models with different parameters, you can follow the following:

```
python tsai_trial.py --preprocessing=multi_3 --arch=MLP --epochs=500 --subject_size=3 --save_output=not_save
```

Please note that you should select a low number of subject_size because training models for more subjects takes too long. If you are looking for reproducing the results for the entire set of subjects, you should set ``` --subject_size=100 ```. For setting the number of pulses per ECG record, you can adjust ``` --preprocessing=multi_2 ``` for 2 pulses per ECG record or ``` --preprocessing=naive ``` for a single pulse per ECG record.

You can run the following command to train the VisGIN model:

```
python VisGIN.py
```

NOTE: Please pay attention line 80 & 81 for the subject size. By default, it is 100 because the dataset consists of 100 subjects. However, it requires long training time (approx. 15 hours). If you are looking for rapid results, you can reduce the subject size by changing 100 to a lower number such as 3 as in baseline models.

## Visualizing results

To visualize the results, you can run ``` python visualize.py ```. If you'd like to see the results of a particular model, please change the directory in line 13. In default, it shows the results of FCN when 3 pulses are being used.

![One-by-one classification results when number of extracted pulses is 3](https://github.com/AslantheAslan/visibility-gin/blob/main/visualizations/res_three_pulse.jpg)

Please note that ```output```  folder holds the previously recorded results for classification when 3 pulses were being used. We also provide results for training with 1 pulse and 2 pulses in folder  ```all_results```. If you want to see the results for these experiments, you can copy the .txt files to ```output``` folder and run visualize.py.
