'''
file for checking no duplicates across splits
'''
import os
import pandas as pd


def check_splits(splits):
    '''
    inputs: tuple of 3 lists of file names
    output: bool whether there is cross over or not
    '''
    seen = []
    repeated = []
    for split in splits:
        for i in split:
            for j in i.split('-'):

                if j in seen:
                    repeated.append(j)
                else:
                    seen.append(j)
    print(len(repeated)==0, repeated)


def read_lists(path):
    '''
    input: path to a txt file
    output: list of file paths, no labels
    '''
    df = pd.read_csv(path, delimiter=' ', header=None)
    files = df[0].to_list()
    return files


datasets = ['coswara', 'epfl']
for dataset in datasets:

    path = os.path.join('/vol/bitbucket/hgc19/covid_cross_datasets/content', dataset, 'splits')

    train = os.path.join(path, 'list_'+str(dataset)+'_train.txt')
    val = os.path.join(path, 'list_'+str(dataset)+'_val.txt')
    test = os.path.join(path, 'list_'+str(dataset)+'_test.txt')

    train_list = read_lists(train)
    val_list = read_lists(val)
    test_list = read_lists(test)


    check_splits((train_list, val_list, test_list))

