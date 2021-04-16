from sklearn.metrics import confusion_matrix
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion(ys, y_hts):
    '''
    function which returns figure of confusion matrix
    '''
    confusion_array = confusion_matrix(ys, y_hts)
    df_cm = pd.DataFrame(confusion_array, index = ['Negative', 'Postive'],
                  columns = ['Negative', 'Positive'])
    fig, ax = plt.subplots(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actual')
    return fig


if __name__ == '__main__':
    ys = [1,0,1]
    yhts = [1,0,1]

    fig = plot_confusion(ys, yhts)
    fig.savefig('experiment.png')
