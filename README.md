# Visibility-GNN
This repository includes the Python implementation of a biometric authentication scheme, (VisGNN:  Visibility Graph Neural Network on One-dimensional Data for Biometric Authentication), using the proposed graph ML method.

## Baseline Models

The following can be run to check baseline models:

```
python tsai_trial.py
```

If you want to reproduce results for different models with different parameters, you can follow the following:

```
python tsai_trial.py --preprocessing=multi_3 --arch=MLP --epochs=500 --subject_size=3 --save_output=not_save
```

Please note that you should select a low number of subject_size because training models for more subjects takes too long. If you are looking for reproducing the results for the entire set of subjects, you should set ``` --subject_size=100 ```. For setting the number of pulses per ECG record, you can adjust ``` --preprocessing=multi_2 ``` for 2 pulses per ECG record or ``` --preprocessing=naive ``` for 2 pulses per ECG record.

