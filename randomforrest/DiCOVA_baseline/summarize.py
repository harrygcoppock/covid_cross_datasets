#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021

@author: Team DiCOVA, IISC, Bangalore
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os,sys

folname=sys.argv[1]

R=[]
for i in range(5):
	res = pickle.load(open(folname+"/fold_{}/val_results.pkl".format(i+1),'rb'))
	R.append(res)

# Plot ROC curves
clr_1 = 'tab:green'
clr_2 = 'tab:green'
clr_3 = 'k'
data_x, data_y, data_auc = [],[],[]
for i in range(5):
	data_x.append(R[i]['FPR'].tolist())
	data_y.append(R[i]['TPR'].tolist())
	data_auc.append(R[i]['AUC']*100)
	plt.plot(data_x[i],data_y[i],label='V-'+str(i+1)+', auc='+str(np.round(data_auc[i],2)), c=clr_1,alpha=0.2)
data_x = np.array(data_x)
data_y = np.array(data_y)
plt.plot(np.mean(data_x,axis=0),np.mean(data_y,axis=0),label='AVG, auc='+str(np.round(np.mean(np.array(data_auc)),2)), c=clr_2,alpha=1,linewidth=2)
plt.plot([0,1],[0,1],linestyle='--', label='chance',c=clr_3,alpha=.5)
plt.legend(loc='lower right', frameon=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
plt.text(0,1,'PATIENT-LEVEL ROC',color='gray',fontsize=12)

plt.gca().set_xlabel('FALSE POSITIVE RATE')
plt.gca().set_ylabel('TRUE POSITIVE RATE')
plt.savefig(os.path.join(folname,'val_roc_plot.pdf'), bbox_inches='tight')
plt.close()


sensitivities = [R[i]['sensitivity']*100 for i in range(5)]
specificities = [R[i]['specificity']*100 for i in range(5)]

with open(os.path.join(folname,'val_summary_metrics.txt'),'w') as f:
	f.write("Sensitivities: "+" ".join([str(round(item,2)) for item in sensitivities])+"\n")
	f.write("Specificities: "+" ".join([str(round(item,2)) for item in specificities])+"\n")
	f.write("AUCs: "+" ".join([str(round(item,2)) for item in data_auc])+"\n")
	f.write("Average sensitivity: "+str(np.round(np.mean(np.array(sensitivities)),2))+" standard deviation:"+str(np.round(np.std(np.array(sensitivities)),2))+"\n")
	f.write("Average specificity: "+str(np.round(np.mean(np.array(specificities)),2))+" standard deviation:"+str(np.round(np.std(np.array(specificities)),2))+"\n")
	f.write("Average AUC: "+str(np.round(np.mean(np.array(data_auc)),2))+" standard deviation:"+str(np.round(np.std(np.array(data_auc)),2))+"\n")
