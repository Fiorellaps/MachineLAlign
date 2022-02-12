<p align="center">
  <img src="https://github.com/Fiorellaps/MachineLAlign/blob/73d6997d4ae171b8ad6a4cdee6a49f50ec9b023f/resources/MachineLalign.PNG" alt="MachineLalign">
</p>

# Aligning Multiple Sequences with Machine Learning models

MachineLalign is a final thesis project developed in Python that includes a web page designed with HTML, CSS and Flask. This study has the goal to prove that DL architectures are an effective approach to MSA, thus, we have created other Machine Learning models, such as Gradient Boosting or Random Forest, to demonstrate that fact. 

Publication available in [Deep Learning and its applications in Multiple Sequence Alignment](https://riuma.uma.es/xmlui/handle/10630/23382)

# Requirements

To be able to compute the sequence alignments, the following tools must be installed and moved to `/MSA_tools` folder:

- [Muscle](https://www.drive5.com/muscle/downloads.htm)
- [MAFFT](https://mafft.cbrc.jp/alignment/software/)
- [Clustal W](http://www.clustal.org/clustal2/)
- [T_coffee](http://www.tcoffee.org/Projects/tcoffee/workshops/tcoffeetutorials/installation.html)

It can be run on Linux or macOS.

# Features

It includes the training and testing of the following models:

- Decision Tree
- Random Forest
- Gradient Boosting
- Adaboost
- K Nearest Neighbours
- CNN-BiLSTM
- CNN

These models have passed through hyperparameter tuning and cross validation.

# Downloading

To download MachineLalign you must colne this Git repository with the following url:


`
$ git clone https://github.com/Fiorellaps/MachineLAlign.git
`


Before running the tool, you must ensure that you have all the packages installed or just execute the following command line:


`
$ pip install -r requirements.txt
`


# Tune models
Firstly, make sure you have downloaded and saved the [input data](https://drive.google.com/file/d/1HzaPY2iEIz_rKM5umXbQzegZ4MZ70qxT/view?usp=sharing) into `/resources`. Then, to apply hyperparametrization tuning with the models got to `/src` and write:


`
$ python3 train_models.py
`


# Test Final models
The final models can be train and tested by the following command line:


`
$ python3 test_models.py
`


Final models will be saved at `/src/models`

# Align Sequences
To compute alignment it is necessary to have tools installed and models saved at `/src/models`(models can be also downloaded [here](https://drive.google.com/drive/folders/10U3TkyHVMklWtd58UiN8gukJdVRNlbBf?usp=sharing)). Go to `/src` and type:


`
$ python3 align_sequences.py 'SEQUENCES_FILE_NAME' 'MODEL_NAME'
`


*Models' names are: 'decisiontree', 'randomforest', 'adaboost', 'gradientboosting', 'knn', 'cnnbilstm', 'cnn'*.

Example:
`
$ python3 align_sequences.py './input_fasta/BAliBASE/BB11001' 'knn'
`


The output will be saved at `/src/output_aligned'`.

# Run the API Rest

The we application can be run going to `/webpage` and writing:


`
$ python3 application.py
`


Once runned you must go to 'Align' section and it will appear an interface like this:

<p align="center">
  <kbd>
  <img src="https://github.com/Fiorellaps/MachineLAlign/blob/ae90d7e9c346d3bff915360209684b9a28b616dc/resources/align_interface.PNG" alt="Alignment interface">
  </kbd>
</p>

Finally, the output will be showed one you introduce the sequence and select one or more models:
<p align="center">
  <kbd>
  <img src="https://github.com/Fiorellaps/MachineLAlign/blob/ae90d7e9c346d3bff915360209684b9a28b616dc/resources/result_interface.PNG" alt="Result interface">
  </kbd>
</p>

For scoring the alignment we have used [pyMSA](https://github.com/benhid/pyMSA/blob/master/README.md).

# Authors
- Student

[Fiorella Piriz Sapio: fiorellapiriz@uma.es](mailto:fiorellapiriz@uma.es)


- Tutors

[Antonio J. Nebro: antonio@lcc.uma.es](mailto:antonio@lcc.uma.es)


[José Manuel García Nieto: jnieto@lcc.uma.es](mailto:jnieto@lcc.uma.es)
