import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

temp = np.array([[72.5,51.2,55.2], [55.4,67.9,46.4], [51.8,39.5,56.6], [71.7, 67.3, 53.1]])
fig = plt.subplots(figsize=(7,6.5))
ax = plt.subplot(1,1,1)
sns.set(font_scale=1.0)#for label size
FS = 12
x_label = ['D-1', 'D-2','D-3']
y_label = ['D-1', 'D-2', 'D-3', 'all']

sns.heatmap(temp, annot=True, fmt='.3g', cmap='Blues', annot_kws={"size": 16},\
            cbar_kws={'label': 'AVG. AUC'})# font size
ax.set_xticks(np.arange(len(x_label))+.5)
ax.set_yticks(np.arange(len(y_label))+.5)
ax.set_xticklabels(x_label,rotation=0,fontsize=FS)
ax.set_yticklabels(y_label,rotation=90,fontsize=FS)
ax.figure.savefig("avg_auc.pdf", bbox_inches='tight')
# plt.show()
sns.set() # Use seaborn's default style to make attractive graphs
sns.set_style("white")
sns.set_style("ticks")

plt.savefig('blah.png')