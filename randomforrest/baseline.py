import os 
import time
import glob
import pickle
import librosa
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
  metadata.set_index('file')
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
  given a path to an audio file, load it as a numpy array
  '''
  file_data = open(path,'rb')
  data = pickle.load(file_data)
  return data.values

def load_data(file_list):
  '''
  Given a dataframe loads into numpy arrays train and test data
  '''
  data = []
  labels = []
  for i in file_list['file']:
    labels.append(load_data(file_list, i))
    audio_path = os.path.join(DIRECTORIES[file_list.loc[i, 'dataset']], audio_path)
    data.append(load_audio(audio_path))
  
  return np.stack(data), np.stack(labels)


    scale = StandardScaler().fit(train_X[dataset_type])
    train_X_stand = scale.transform(train_X[dataset_type])
    # clf = RandomForestClassifier(max_depth=6, random_state=0, criterion='gini').fit(train_X_stand, train_Y[dataset_type])
    clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0, n_jobs=-1).fit(train_X_stand, train_Y[dataset_type])
    # clf = RandomForestClassifier(max_depth=24, min_samples_leaf=3, n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True, random_state=0).fit(train_X[dataset_type], train_Y[dataset_type])
    for key in dataset_types:
      val_X_stand = scale.transform(val_X[key])
      output_scores = clf.predict_proba(val_X_stand)[:,1]
      tp, fp = compute_tp_fp(output_scores, val_Y[key])
      val_auc[dataset_type+'_'+key] = auc(fp, tp)
      loop_val_auc[iter][i].append(val_auc[dataset_type+'_'+key])
    i = i + 1

  # pooled evaluation
  train_X_all = np.vstack((train_X['cambridge'], train_X['dicova'], train_X['epfl']))
  train_Y_all = np.hstack((train_Y['cambridge'], train_Y['dicova'], train_Y['epfl']))

  loop_val_auc[iter].append([])
  scale = StandardScaler().fit(train_X_all)
  train_X_stand = scale.transform(train_X_all)
  clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0, n_jobs=-1).fit(train_X_stand, train_Y_all)
  for key in dataset_types:
    val_X_stand = scale.transform(val_X[key])
    output_scores = clf.predict_proba(val_X_stand)[:,1]
    tp, fp = compute_tp_fp(output_scores, val_Y[key])
    val_auc['pooled'+'_'+key] = auc(fp, tp)
    loop_val_auc[iter][i].append(val_auc['pooled'+'_'+key])

# to train a system on cambridge and test on the cambridge only
dataset_type = 'cambridge'
train_X = {}
train_Y = {}

val_X = {}
val_Y = {}

train_X[dataset_type], val_X[dataset_type], train_Y[dataset_type], val_Y[dataset_type] = \
            make_train_val_split(dataset_type, file_list)
scale = StandardScaler().fit(train_X[dataset_type])
train_X_stand = scale.transform(train_X[dataset_type])
clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0, n_jobs=-1).fit(train_X_stand, train_Y[dataset_type])

test_X_stand = scale.transform(test_X)
output_scores = clf.predict_proba(test_X_stand)[:,1]

plt.plot(output_scores)

plt.plot(output_scores)

plt.plot(output_scores)

df = {}
df['filename'] = []
df['prediction'] = []
thres = 0.3
for i in range(len(output_scores)):
  df['filename'].append(test_list[i].split('/')[-1].split('.p')[0])
  if output_scores[i] >thres:
    df['prediction'].append('positive')
  else:
    df['prediction'].append('negative')

new = pd.DataFrame.from_dict(df)
new_1 = new.sort_values(by = 'filename')
new_1.to_csv("cambridge_test_scores.csv", index=False)

new_1

# plot decisions at 80% sensitivity
temp = np.array(loop_val_auc)
fig = plt.subplots(figsize=(7,6.5))
ax = plt.subplot(1,1,1)
sns.set(font_scale=1.0)#for label size
FS = 12
x_label = ['D-1', 'D-2','D-3']
y_label = ['D-1', 'D-2', 'D-3', 'D-All']

sns.heatmap(np.mean(temp,axis=0)*100, annot=True, fmt='.3g', cmap='Blues', annot_kws={"size": 16},\
            cbar_kws={'label': 'AVG. AUC'})# font size
ax.set_xticks(np.arange(len(x_label))+.5)
ax.set_yticks(np.arange(len(y_label))+.5)
ax.set_xticklabels(x_label,rotation=0,fontsize=FS)
ax.set_yticklabels(y_label,rotation=90,fontsize=FS)
ax.figure.savefig("avg_auc.pdf", bbox_inches='tight')
plt.show()
sns.set() # Use seaborn's default style to make attractive graphs
sns.set_style("white")
sns.set_style("ticks")

# plot decisions at 80% sensitivity
temp = np.array(loop_val_auc)
fig = plt.subplots(figsize=(7,6.5))
ax = plt.subplot(1,1,1)
sns.set(font_scale=1.0)#for label size
FS = 12
x_label = ['D-1', 'D-2','D-3']
y_label = ['D-1', 'D-2', 'D-3', 'D-All']

sns.heatmap(np.std(temp,axis=0)*100, annot=True, fmt='.3g', cmap='Blues', annot_kws={"size": 16},\
            cbar_kws={'label': 'AVG. AUC'})# font size
ax.set_xticks(np.arange(len(x_label))+.5)
ax.set_yticks(np.arange(len(y_label))+.5)
ax.set_xticklabels(x_label,rotation=0,fontsize=FS)
ax.set_yticklabels(y_label,rotation=90,fontsize=FS)
ax.figure.savefig("avg_auc.pdf", bbox_inches='tight')
plt.show()
sns.set() # Use seaborn's default style to make attractive graphs
sns.set_style("white")
sns.set_style("ticks")

# train and cross-evaluate across datasets
val_auc = {}
for dataset_type in dataset_types:
  scale = StandardScaler().fit(train_X[dataset_type])
  train_X_stand = scale.transform(train_X[dataset_type])
  # clf = RandomForestClassifier(max_depth=6, random_state=0, criterion='gini').fit(train_X_stand, train_Y[dataset_type])
  clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0).fit(train_X_stand, train_Y[dataset_type])
  # clf = RandomForestClassifier(max_depth=24, min_samples_leaf=3, n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True, random_state=0).fit(train_X[dataset_type], train_Y[dataset_type])
  for key in dataset_types:
    val_X_stand = scale.transform(val_X[key])
    output_scores = clf.predict_proba(val_X_stand)[:,1]
    tp, fp = compute_tp_fp(output_scores, val_Y[key])
    val_auc[dataset_type+'_'+key] = auc(fp, tp)

# pooled evaluation
train_X_all = np.vstack((train_X['cambridge'], train_X['dicova'], train_X['epfl']))
train_Y_all = np.hstack((train_Y['cambridge'], train_Y['dicova'], train_Y['epfl']))

train_X_stand = scale.transform(train_X_all)
clf = RandomForestClassifier(max_depth=24, n_estimators=257, criterion='entropy', random_state=0).fit(train_X_stand, train_Y_all)
for key in dataset_types:
  val_X_stand = scale.transform(val_X[key])
  output_scores = clf.predict_proba(val_X_stand)[:,1]
  tp, fp = compute_tp_fp(output_scores, val_Y[key])
  val_auc['pooled'+'_'+key] = auc(fp, tp)

for i in range(4):
  for j in range(3):

val_auc

val_auc

plt.plot(clf.feature_importances_)

plt.plot(clf.feature_importances_)



https://osf.io/k8t23/download