

import glob
import zipfile
import os

# DOWNLOAD DATASETS
# coswara cough zip FLAC dataset 
print('Downloading COSWARA cough ZIP ...')
os.system('wget https://osf.io/67gfh/download -P /vol/bitbucket/hgc19/covid_cross_datasets/content/coswara_cough_download/')
print('Downloading zip ... complete')
# unzip dataset 
print('Unzipping ... started')
file_name = '/vol/bitbucket/hgc19/covid_cross_datasets/content/coswara_cough_download/download'
extract_path = '/vol/bitbucket/hgc19/covid_cross_datasets/content/coswara/'
with zipfile.ZipFile(file_name) as file:
    file.extractall(path = extract_path)
print('Unzipping ... complete')

# epfl cough zip FLAC dataset 
print('Downloading COSWARA cough ZIP ...')
os.system('wget https://osf.io/zasf3/download -P /vol/bitbucket/hgc19/covid_cross_datasets/content/epfl_cough_download/')
print('Downloading zip ... complete')

# unzip dataset 
print('Unzipping ... started')
file_name = '/vol/bitbucket/hgc19/covid_cross_datasets/content/epfl_cough_download/download'
extract_path = '/vol/bitbucket/hgc19/covid_cross_datasets/content/epfl/'
with zipfile.ZipFile(file_name) as file:
    file.extractall(path = extract_path)
print('Unzipping ... complete')


# CREATE DICTIONARY with pos/neg lists
dataset_types = ['coswara','epfl']
# make train pos neg list for each type
file_list = {}
for key in dataset_types:
  if key == 'coswara':
    path_audio = '/content/coswara/cough-heavy/'
    file_list[key+'_pos'] = glob.glob(path_audio + '*_pos*.flac')
    file_list[key+'_neg'] = glob.glob(path_audio + '*_neg*.flac')
  if key == 'epfl':
    path_audio = '/content/epfl/'
    file_list[key+'_pos'] = glob.glob(path_audio + '*_pos*.flac')
    file_list[key+'_neg'] = glob.glob(path_audio + '*_neg*.flac')

# PRINT COUNT 
for key in file_list:
  print('Dataset '+key+ ': '+str(len(file_list[key])))