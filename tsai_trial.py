
"""
In this script, we evaluated performances of benchmark models. TSAI library has been used for all model definitions
in this part. More description is available in the github repository.

"""

from tsai.all import *
from tsai.inference import load_learner
import time

from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix

computer_setup()

import os

import argparse


def main(args):
  directory = os.getcwd()
  save_directory = directory + "\output"

  if args.preprocessing == "naive":
    read_directory = directory + "\data\ECG\combined_pulses.txt"
  elif args.preprocessing == "baseline_corrected":
    read_directory = directory + "\data\ECG\corrected_baseline_pulses.txt"
  elif args.preprocessing == "multi_2":
    read_directory = directory + "\data\ECG\multi_patients_2.txt"
  elif args.preprocessing == "multi_3":
    read_directory = directory + "\data\ECG\multi_patients_3.txt"

  with open(read_directory,"rb") as fp:   # Unpickling
    combined_pulses = pickle.load(fp)


  print(len(combined_pulses), len(combined_pulses[0]), len(combined_pulses[0][0]))
  test_accuracy_tuples = []
  train_accuracy_tuples = []

  ### select either corr_base_pulses or combined_pulses, and change them in the training loop accordingly.

  for i in tqdm(range(args.subject_size)):
    for j in tqdm(range(i+1,args.subject_size)):
      genuine = i
      impostor = j

      X1 = np.append(np.array(combined_pulses[genuine]), np.array(combined_pulses[impostor]), axis=0)
      y1 = np.concatenate((np.zeros(np.array(combined_pulses[genuine]).shape[0]), np.ones(np.array(combined_pulses[impostor]).shape[0])))

      split_max_samp = np.array(combined_pulses[genuine]).shape[0] + np.array(combined_pulses[impostor]).shape[0]

      splits1 = get_splits(np.random.randint(0,1,split_max_samp), shuffle=True, stratify=False, valid_size=args.test_split, show_plot=False)
      K = 1

      res = []
      subl = []
      cnt = 0
      for sub in X1:
        subl.append(sub)
        cnt = cnt + 1
        if cnt >= K:
            res.append(subl)
            subl = []
            cnt = 0

      X1_3d = np.array(res)
      batch_tfms = TSStandardize()

      roc_auc = RocAucBinary(average='micro', max_fpr=1.0)
      clf = TSClassifier(X1_3d, y1, splits=splits1, path='models', arch=args.arch, batch_tfms=batch_tfms, metrics=accuracy, train_metrics=True)
      clf.fit_one_cycle(args.epochs, args.learning_rate)
      # clf.export("clf.pkl")

      # clf = load_learner("models/clf.pkl")
      _, target, preds1 = clf.get_X_preds(X1_3d[splits1[1]], y1[splits1[1]])
      _, target_train, preds_train = clf.get_X_preds(X1_3d[splits1[0]], y1[splits1[0]])


      list3 = []
      for k2 in range(len(preds_train)):
        t2 = float(preds_train[k2])
        list3.append(t2)
      print("Genuine patient number:", genuine, "impostor patient number:", impostor)
      print("Training accuracy:", accuracy_score(y1[splits1[0]], list3))

      train_fp = confusion_matrix(y1[splits1[0]], list3)[0][1]
      train_fn = confusion_matrix(y1[splits1[0]], list3)[1][0]

      train_tp = confusion_matrix(y1[splits1[0]], list3)[0][0]
      train_tn = confusion_matrix(y1[splits1[0]], list3)[1][1]

      train_accuracy_tuples += [(genuine, impostor, accuracy_score(y1[splits1[0]], list3), train_tp, train_fp, train_fn, train_tn)]


      list2 = []
      for k in range(len(preds1)):
        t = float(preds1[k])
        list2.append(t)
      print("Genuine patient number:",genuine,"impostor patient number:",impostor)
      print( "Test accuracy:", accuracy_score(y1[splits1[1]], list2))

      fp = confusion_matrix(y1[splits1[1]], list2)[0][1]
      fn = confusion_matrix(y1[splits1[1]], list2)[1][0]

      tp = confusion_matrix(y1[splits1[1]], list2)[0][0]
      tn = confusion_matrix(y1[splits1[1]], list2)[1][1]

      test_accuracy_tuples += [(genuine, impostor, accuracy_score(y1[splits1[1]], list2), tp, fp, fn, tn)]


  print(train_accuracy_tuples)
  print(test_accuracy_tuples)


  if args.save_output == "save":
    with open(save_directory + "\\" + args.arch + "_accuracy.txt", "wb") as fp:   #Pickling
      pickle.dump(test_accuracy_tuples, fp) # save the accuracy scores per match

    with open(save_directory + "\\" + args.arch + "_train_accuracy.txt", "wb") as fp:   #Pickling
      pickle.dump(train_accuracy_tuples, fp) # save the accuracy scores per match
    print("Output saved!")

  elif args.save_output == "not_save":
    print("Output is not saved!")



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optional arguments for your further experiments')
  ## For environment costruction
  parser.add_argument("--preprocessing", type=str, default="naive",
                      help="the preprocessing method on the data")
  parser.add_argument("--arch", type=str, default="ResCNN")
  parser.add_argument("--test_split", type=float, default=0.2)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--learning_rate", type=float, default=3e-4)
  parser.add_argument("--subject_size", type=int, default=3, help="The full dataset consists of 100 patients but you "
                                                                  "may like to train the model with few patients for "
                                                                  "rapid results")
  parser.add_argument("--save_output", type=str, default="save")
  args = parser.parse_args()
  print(args)

  main(args)