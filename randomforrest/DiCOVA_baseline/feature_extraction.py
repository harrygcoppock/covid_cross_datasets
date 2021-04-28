#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021

@author: Team DiCOVA, IISC, Bangalore
"""
import argparse, configparser
import pickle
import librosa
import numpy as np

def compute_SAD(sig,config):
	# Speech activity detection based on sample thresholding
	# Expects a normalized waveform as input
	# Uses a margin of at the boundary
    fs = int(config['default']['sampling_rate'])
    sad_thres = float(config['sad']['threshold'])
    sad_start_end_sil_length = int(int(config['sad']['start_end_sil_length'])*1e-3*fs)
    sad_margin_length = int(int(config['sad']['silence_margin'])*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>sad_thres] = 1
    sad = np.zeros(sig.shape)
    for i in range(len(sample_activity)):
        if sample_activity[i] == 1:
            sad[i-sad_margin_length:i+sad_margin_length] = 1
    sad[0:sad_start_end_sil_length] = 0
    sad[-sad_start_end_sil_length:] = 0
    return sad


def read_audio(file_path,sampling_rate):
	fs=librosa.get_samplerate(file_path)
	try:
		s,_ = librosa.load(file_path,sr=sampling_rate)
		if np.mean(s)==0 or len(s)<1024:
			raise ValueError()
		# waveform level amplitude normalization
		s = s/np.max(np.abs(s))
	except ValueError:
		s = None
		print("Read audio failed for "+file_path)		
	return s

def compute_mfcc(s,config):
	# Compute MFCC using librosa toolkit. 
	F = librosa.feature.mfcc(s,sr=int(config['default']['sampling_rate']),
								n_mfcc=int(config['mfcc']['n_mfcc']),
								n_fft = int(config['default']['window_size']),
								hop_length = int(config['default']['window_shift']),
								n_mels = int(config['mfcc']['n_mels']),
								fmax = int(config['mfcc']['fmax']))

	features = np.array(F)
	if config['mfcc']['add_deltas'] in ['True','true','TRUE','1']:
		deltas = librosa.feature.delta(F)
		features = np.concatenate((features,deltas),axis=0)

	if config['mfcc']['add_delta_deltas'] in ['True','true','TRUE','1']:
		ddeltas = librosa.feature.delta(F,order=2)
		features = np.concatenate((features,ddeltas),axis=0)

	return features

def main(config, in_wav_list, out_folder):

	in_wav_list = open(in_wav_list).readlines()
	in_wav_list = [line.strip().split(" ") for line in in_wav_list]

	feats_list=[]
	for file_id,file_path in in_wav_list:
		s = read_audio(file_path,int(config['default']['sampling_rate']))

		if s is None:
			continue
	
		sad = compute_SAD(s,config)
		ind=np.where(sad==1)
		s = s[ind]	# Equivalent to stripping the silence portions of the waveform
		if len(s)<1024:
			continue

		if config['default']['feature_type'] == 'mfcc':
			f = compute_mfcc(s,config)
		else:
			raise ValueError("Need to implement the feature: "+config['default']['feature_type'])

		out_file_name = out_folder+"/"+file_id+'_'+config['default']['feature_type']+'.pkl'
		with open(out_folder+"/"+file_id+'_'+config['default']['feature_type']+'.pkl','wb') as fp:
			pickle.dump(f,fp)
		feats_list.append((file_id,out_file_name))
	with open(out_folder+"/feats.scp","w") as f:
		for file_id,out_file_name in feats_list:
			f.write(file_id+" "+out_file_name+"\n")	

	print("Feature extraction completed for "+str(len(in_wav_list))+" files")
	print("Feature matrices saved to "+out_folder)
	
if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config','-c',required=True)
	parser.add_argument('--in_wav_list','-i',required=True)
	parser.add_argument('--out_folder','-o',required=False)
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)
	main(config, args.in_wav_list, args.out_folder)
