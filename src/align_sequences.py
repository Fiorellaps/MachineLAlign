"""
@author fiorellaps
@since 05/05/21
Program for aligning sequences from command line

"""
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os, sys
from pandas.core.frame import DataFrame
from datetime import date

seqs_sys = 'ARNDCQEGHILKMFPSTWYV-'
todayDate = str(date.today())

def compute_alignment(family_classification, model_name, file_name):
    command_line=''
    print(file_name)
    if family_classification == 0:
        out_file_name = 'output_aligned/' + todayDate + '_' + model_name + '_tcoffee'
        command_line = 'mafft --auto --inputorder --quiet '+ file_name + ' > ' + out_file_name
    elif family_classification == 1:
        out_file_name = 'output_aligned/' + todayDate + '_' + model_name + '_mafft'
        command_line = '../MSA_tools/muscle3.8.31_i86linux64 -in '+ file_name + ' -fastaout ' + out_file_name + ' -quiet'
    elif family_classification == 2:
        out_file_name = 'output_aligned/' + todayDate + '_' + model_name + '_muscle'
        command_line = '../MSA_tools/clustalw2 -infile=' + file_name + ' -outfile=' + out_file_name + ' -output=FASTA -ALIGN -QUIET -OUTORDER=input'
    elif family_classification == 3:
        out_file_name = 'output_aligned/' + todayDate + '_' + model_name + '_clustalw2'
        command_line = 't_coffee ' + file_name + ' -output fasta -outfile=' + out_file_name
    os.system(command_line)
    print('DLPAlign Finished!')
    

def findMaxClassification(np_arr, lens, model_name, file_name):
    list_sys = [0.22, 0.17, 0.21, 0.4]
    max = 0
    max_classification = 0
    for i in range(len(np_arr)):
        if (np_arr[i] / lens) / list_sys[i] > max:
            max_classification = i
            max = (np_arr[i] / lens) / list_sys[i]
    return compute_alignment(max_classification, model_name, file_name)

def classifyFamily(seqs, model, model_name, file_name):
    df = DataFrame({'Seqs': seqs})
    df.reset_index()
    seqs_ = df.Seqs.values
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(seqs_sys)
    X = tokenizer.texts_to_sequences(seqs_)
    max_length = 1064
    X = sequence.pad_sequences(X, maxlen=max_length)
    
    if model_name == 'cnnbilstm' or  model_name == 'cnn':
        np_list = np.argmax(model.predict(X), axis = 1)
        
    else :
        np_list = model.predict(X)

    return findMaxClassification(np.bincount(np_list), len(np_list), model_name, file_name)


def readModel(model_name):

    if(model_name == "adaboost"):
        model = joblib.load('../src/models/model_AdaBoostClassifier.pkl')
    elif(model_name == "knn"):
        model = joblib.load('../src/models/model_KNeighborsClassifier.pkl')
    elif(model_name == "decisiontree"):
        model = joblib.load('../src/models/model_DecisionTreeClassifier.pkl')
    elif(model_name == "gradientboosting"):
        model = joblib.load('../src/models/model_GradientBoostingClassifier.pkl')
    elif(model_name == "randomforest"):
        model = joblib.load('../src/models/model_RandomForestClassifier.pkl')
    elif(model_name == "cnnbilstm"):
        model = load_model('../src/models/model_CNN_BiLSTM.h5')
    elif(model_name == "cnn"):
        model = load_model('../src/models/model_CNN.h5')
    return model

def readSequencesFromFile(family_file):
    sequences = []
    temporal_value = ""
    sequences_pairs = []
    file_in = open(family_file, 'r')
    file_processed = file_in.read().splitlines()
    file_in.close()
    for line in file_processed:
        if len(temporal_value) > 0 and '>' in line:
            sequences.append(temporal_value)
            temporal_value = ""
        elif '>' not in line:
            temporal_value += line
    if len(temporal_value) > 0:
        sequences.append(temporal_value)
    for i in range(len(sequences) - 1):
        for j in range(i, len(sequences)):
            sequences_pairs.append(sequences[i] + '-' + sequences[j])
    return sequences_pairs

def alignSequencesFromFile(file_name, input_model):
    model = readModel(input_model)
    sequences_pairs = readSequencesFromFile(file_name)
    return classifyFamily(sequences_pairs, model, input_model, file_name)


if __name__ == '__main__':
    '''
    EXAMPLE CALL:
    python3 align_sequences.py './input_fasta/BAliBASE/BB11001' 'knn'
    '''
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
    else:
        alignSequencesFromFile(sys.argv[1], sys.argv[2])
