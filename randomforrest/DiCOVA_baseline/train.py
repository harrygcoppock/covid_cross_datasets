#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021

@author: Team DiCOVA, IISC, Bangalore
"""
import argparse, configparser
import pickle, random
import numpy as np
from models import *

def get_data(file_list,feats_file,labels_file,shuffle=False):

	#%% read the list of files
	file_list = open(file_list).readlines()
	file_list = [line.strip().split() for line in file_list]

	#%% read labels
	temp = open(labels_file).readlines()
	temp = [line.strip().split() for line in temp]
	labels={}
	categories = ['n','p']
	for fil,label in temp:
		labels[fil]=categories.index(label)
	del temp

	#%% read feats.scp
	temp = open(feats_file).readlines()
	temp = [line.strip().split() for line in temp]
	feats={}
	for fil,filpath in temp:
		feats[fil]=filpath
	del temp

	#%% make examples
	egs = []
	for fil,_ in file_list:
		if feats.get(fil,None):
			F = pickle.load(open(feats[fil],'rb'))
			label = labels.get(fil,None)
			if label is not None:
				egs.append( np.concatenate( (np.array(F.T),np.array([label]*F.shape[1]).reshape(F.shape[1],1)),axis=1 ) )

	egs = np.vstack(egs)

	if shuffle:
		np.random.shuffle(egs)	
	return egs[:,:-1],egs[:,-1]



def main(config,datadir,outdir):

	# Training dataset
	train_feats,train_labels = get_data(datadir+"/train.scp",datadir+"/feats.scp",datadir+"/train_labels", shuffle=True)

	model_type = config['default']['model']
	verbose = True if config[model_type]['verbose'] in ['True','TRUE','true','1'] else False
	cw = config[model_type]['class_weight']
	cw = cw if 'balanced' in cw else None
	if model_type == 'LogisticRegression':

		model_args = { 'c':int(config[model_type]['C']), 
						'max_iter':int(config[model_type]['max_iter']), 
						'solver':config[model_type]['solver'],
						'penalty':config[model_type]['penalty'], 
						'class_weight':cw,
						'verbose':verbose,
						'random_state':int(config['default']['seed'])}
		model = LR(model_args)
		model.run_fit(train_feats,train_labels)

	elif model_type == 'RandomForest':

		model_args = { 'n_estimators':int(config[model_type]['n_estimators']), 
						'class_weight':cw,
						'verbose': verbose,
						'random_state':int(config['default']['seed'])}
		model = RF(model_args)
		model.run_fit(train_feats,train_labels)

	elif model_type == 'MLP':

		if cw == 'balanced':
			train_data = np.concatenate((train_feats,train_labels.reshape(train_feats.shape[0],1)),axis=1)		
			ind = np.where(train_data[:,-1]==1)[0]
			n_positives = len(ind)
			n_negatives=train_data.shape[0]-n_positives
			upsample_factor = int(n_negatives/n_positives)-1			
			for i in range(upsample_factor):
				train_data = np.concatenate((train_data,train_data[ind,:]),axis=0)
			np.random.shuffle(train_data)
			train_feats = train_data[:,:-1]
			train_labels = train_data[:,-1]

		hidden_layer_sizes = [int(item) for item in config[model_type]['hidden_layer_sizes'].split(",")] 
		model_args = {'learning_rate_init':float(config[model_type]['learning_rate_init']), 
						'alpha':float(config[model_type]['alpha']), 
						'solver':config[model_type]['solver'], 
						'hidden_layer_sizes':hidden_layer_sizes, 
						'max_iter':int(config[model_type]['max_iter']),
						'activation':config[model_type]['activation'],
						'verbose': verbose,
						'random_state':int(config['default']['seed'])}
		model = MLP(model_args)
		model.run_fit(train_feats,train_labels)

	else:
		raise ValueError("Implement the model: "+model_type)

	pickle.dump(model,open(outdir+"/model.pkl","wb"))	

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config','-c',required=True)
	parser.add_argument('--datadir','-d',required=True)
	parser.add_argument('--outdir','-o',required=True)	
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)

	np.random.seed(int(config['default']['seed']))
	random.seed(int(config['default']['seed']))

	main(config, args.datadir, args.outdir)
