### Repo for cross evaluating CIdeR and a randomforrest classifier on the datasets: coswara, epfl and compare.
note that this is just for the coughing modality. Cross evaluation refers to training a model on dataset x then evaluating it on dataset y with no fine tuning.

- To train CideR run:
```python
python main.py
```
The args to specify what to train on along with the hyperparameters are in args.txt. For more information please see [CIdeR](https://github.com/glam-imperial/cider).

- To train a random forest:
```python
python randomforest/baseline.py
```
This segements the audio files, extracts MFCCs and classifies each MFCC vector. The mean of each decision (on each MFCC vector) is the final probability prediction of each audio clip.


- Requirements:

```sh
pip install -r requirements.txt
```