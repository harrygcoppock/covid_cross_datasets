import os 
import time
import glob
import pickle
import librosa
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import tqdm
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from multiprocessing import Pool
from functools import partial


from DiCOVA_baseline.feature_extraction import read_audio

sns.set() # Use seaborn's default style to make attractive graphs
sns.set_style("white")
sns.set_style("ticks")


"""
TODO
- train model on train + val
- test model
"""

PATHS = {
    'coswara':'/vol/bitbucket/hgc19/covid_cross_datasets/content/coswara/splits/list_coswara_{dset}.txt',
    'epfl':'/vol/bitbucket/hgc19/covid_cross_datasets/content/epfl/splits/list_epfl_{dset}.txt',
    'compare': '/vol/bitbucket/hgc19/COMPARE_data/cough/lab/{dset}.csv'

}
DIRECTORIES = {
    'coswara':'/vol/bitbucket/hgc19/covid_cross_datasets/content/coswara/cough/',
    'epfl': '/vol/bitbucket/hgc19/covid_cross_datasets/content/epfl/cough/',
    'compare': '/vol/bitbucket/hgc19/COMPARE_data/cough/wav/'
}
def file_paths(split, dataset):
  '''
  function which returns a pandas df of the file paths
  input:
  split: str detailing train, val, test
  dataset: list of strings detailing which datasets to use
  '''
  li = []
  for name, path in PATHS.items():
    if name not in dataset:
      continue
    df = pd.read_csv(path.format(dset=split),
                names=['file', 'label'],
                delimiter=',' if name == 'compare' else ' ',
                skiprows=1 if name == 'compare' else 0)
    df['dataset'] = name
    li.append(df)
  metadata = pd.concat(li, axis=0, ignore_index=True)
  metadata = metadata.set_index('file')
  return metadata

def return_label(file_list, audio_path):
  '''
  given the file name and the list of files returns the label as an int
  '''
  cov_pos = 'positive' if file_list.loc[audio_path, 'dataset'] == 'compare' else 'p'
  cov_neg = 'negative' if file_list.loc[audio_path, 'dataset'] == 'compare' else 'n'
  if file_list.loc[audio_path, 'label'] == cov_pos:
      label = 1
  elif file_list.loc[audio_path, 'label'] == cov_neg:
      label = 0
  else:
      raise f"Error, {self.file_list.loc[audio_path, 'label']} is not a valid category"

  return label

def load_audio(path):
  '''
  given a path to an audio file, extract features and load it as a numpy array
  '''
  # signal, sample_rate = librosa.load(path, sr=48000)
  s = read_audio(path, 48000)
  sad = compute_SAD(s)
  ind = np.where(sad==1)
  s = s[ind]
  F = librosa.feature.mfcc(s,sr=44100,
              n_mfcc=39,
              n_fft=1024,
              hop_length=441,
              n_mels=64,
              fmax=22050)

  features = np.array(F)
  return features.T

def compute_SAD(sig):
	# Speech activity detection based on sample thresholding
	# Expects a normalized waveform as input
	# Uses a margin of at the boundary
    fs = 44100
    sad_thres = 0.0001
    sad_start_end_sil_length = int(20*1e-3*fs)
    sad_margin_length = int(50*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>sad_thres] = 1
    sad = np.zeros(sig.shape)
    for i in range(len(sample_activity)):
        if sample_activity[i] == 1:
            sad[i-sad_margin_length:i+sad_margin_length] = 1
    sad[0:sad_start_end_sil_length] = 0
    sad[-sad_start_end_sil_length:] = 0
    return sad

def paralise(file_list, i):
  label = return_label(file_list, i)
  audio_path = os.path.join(DIRECTORIES[file_list.loc[i, 'dataset']], i)
  data = load_audio(audio_path)

  rep = np.concatenate((data, np.array([label]*data.shape[0]).reshape(data.shape[0], 1)), axis=1)
  return rep[:,:-1], rep[:,-1]

def load_data(file_list):
  '''
  Given a dataframe loads into numpy arrays train and test data
  '''
  data = []
  labels = []
  pool = Pool()
  func = partial(paralise, file_list)
  data, labels = zip(*pool.map(func, file_list.index))

  # for i in file_list.index:
  #   labels.append(return_label(file_list, i))
  #   audio_path = os.path.join(DIRECTORIES[file_list.loc[i, 'dataset']], i)
  #   data.append(load_audio(audio_path))
  
  return np.vstack(data), np.vstack(labels)


def main(dataset):
  '''
  given dataset name, train on this dataset and evaluate on all 3 tests
  input:
  dataset: list of names of datasets
  '''
  print('***Loading Features***')
  train_paths = file_paths('trainval', dataset)
  train_X, train_Y = load_data(train_paths)
  print('***Training Forest***')
  scale = StandardScaler().fit(train_X)
  train_X_stand = scale.transform(train_X)
  clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0, n_jobs=-1).fit(train_X_stand, train_Y)
  print('***Testing Forest***')
  scores = {}
  for test_dataset in tqdm(PATHS.keys()):
    test_paths = file_paths('test', test_dataset)
    test_X, test_Y = load_data(test_paths)
    test_X_stand = scale.transform(test_X)
    output_scores = clf.predict_proba(test_X_stand)[:,1]

    fpr, tpr, _ = roc_curve(test_Y, output_scores)
    roc_auc = auc(fpr, tpr)
    scores[test_dataset] = roc_auc
    score_name = dataset+test_dataset+'scores.txt'
    with open(score_name, 'w') as file:
     file.write(json.dumps(scores))
  


if __name__ == '__main__':
  main('coswara')