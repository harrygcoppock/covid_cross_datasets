
#############################################################################################
#                                                                                           # 
#                                    DiCOVA 2021 Challenge                                  #
#                            Diagnosing COVID-19 Using Acoustics                            #
#                               Track-1 Baseline system software			                #
#                                http://dicova2021.github.io/                               #
#                            (A special session at Interspeech 2021)                        # 
#                               https://www.interspeech2021.org/                            #
#                                                                                           #  
#############################################################################################

---------
1. About:
---------

The DiCOVA Special Session/Challenge is designed to find scientific and engineering insights
into diagnosing COVID-19 using acoustics. As part of the challenge an acoustic dataset gathered
from COVID-19 positive and non-COVID-19 individuals is provided for analysis. Here we provide
a baseline system software to get you started on the challenge. This is a system written with a
python back-end and a shell front-end, and follows object oriented programming. It is composed
of the following parts:
- Feature extraction
- Model initialization
- Model training, and exporting
- Validation data classification
- Performance computation

The modularized structure, organization into seprate python files, and intergration of the
pipeline using shell allows easy scalability and modification as per user needs. As an example,
we have presented a baseline system with
- Features: MFCCs (39 dimensional + delta + delta-delta)
- Classifier: Random Forest, MLP (1 layer), and Logistic Regression
These are simple model to get you started on the challenge. We will like you to build your own
model and beat the performance of the baseline system as best as you can. Below we describe the
directory structure of the content inside the software.

-----------------------
2. Directory structure:
-----------------------
.
├── LICENSE.md
├── Readme
├── conf
│   ├── feature.conf
│   ├── train.conf
│   ├── train_lr.conf
│   ├── train_mlp.conf
│   └── train_rf.conf
├── feature_extraction.py
├── infer.py
├── models.py
├── parse_options.sh
├── REQUIREMENTS.txt
├── run.sh
├── scoring.py
├── summarize.py
└── train.py

----------------------
3. Directory contents:
----------------------

- conf/
	-- feature.conf				[ Configuration file used by the feature extraction module ]
    -- train_lr.conf            [ Configuration file used by LR classifier module ]
    -- train_mlp.conf           [ Configuration file used by MLP classifer module ]
    -- train_rf.conf            [ Configuration file used by RF classifer module ]

- run.sh					    [ Master (shell) script to run the codes ]
- parse_options.sh				[ Facilitates inputing command-line arguments to run.sh (borrowed
                                from Kaldi, note the license details inside it)]

- feature_extraction.py         [ Extract features: requires feature configuration, and the list
                                of files wav.scp ]	

- models.py                     [ Model definition: contains models details]
- train.py                      [ Training models: uses models.py ]
- infer.py                      [ Inference: forward pass through the trained model to generate
                                score as probalities]
- scoring.py                    [ Performance: computes false positives, true positives, etc.,
                                from ground truth labels and the system scores ]
- summarize.py                  [ Summarize: document the results across the folds and generate
                                average performance metrics ]
- REQUIREMENTS.txt              [ Contains a list of dependencies to run the baseline system ]

--------------
4. How to run:
--------------

- Create "DICOVA" directory
- Place "DiCOVA_Train_Val_Data_Release" inside "DICOVA" directory
- Place the "DiCOVA_baseline" inside "DICOVA" directory
- Open shell terminal (Linux), navigate to "DiCOVA_baseline"
- Type the following and hit enter: 
$ ./run.sh

--------------------
4. Baseline results:
--------------------
-------------------------------------------------------------------------------------
[ Specificity is computed at sensitivity >=80% ] 
[ Average metrics across folds are reported below ] 
---------------------------------------------------------------------------------------------------------
Model								|	AUC	(std)	|	Sensitivity (std)			| Specificity (std)	|
---------------------------------------------------------------------------------------------------------
Logistic Regression					|	64.04 (3.42)	|		83.20 (2.99)		|	  37.51	(5.39)	|
---------------------------------------------------------------------------------------------------------
Multilayer Perceptron				|	66.80 (3.05)	|		82.30 (3.20)		|	  37.62	(4.37)	|	
(Single layer, 25 hidden units)		|					|							|					|
---------------------------------------------------------------------------------------------------------
Random Forest*						|	66.96 (2.71)	|		90.40 (6.50)		|	  28.50 (12.71)	|
---------------------------------------------------------------------------------------------------------
* best baseline model
* depending on the version of python packages in your system, the performance may be little different in
the decimal places

--------------
7. Contact Us:
--------------

Please reach out to dicova2021@gmail.com for any queries.


--------------
8. Organizers:
--------------

Team DiCOVA
- Sriram Ganapathy | Assistant Professor, IISc, Bangalore
- Prasanta Kumar Ghosh | Associate Professor, IISc, Bangalore
- Neeraj Kumar Sharma | CV Raman Postdoctoral Researcher, IISc, Bangalore
- Srikanth Raj Chetupalli | Postdoctoral Researcher, IISc, Bangalore
- Prashant Krishnan | Research Assistant, IISc, Bangalore
- Rohit Kumar | Research Associate, IISc, Bangalore
- Ananya Muguli | Research Assistant, IISc, Bangalore

#############################################################################################
