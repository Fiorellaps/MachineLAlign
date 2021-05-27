import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn import preprocessing
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from  tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from  tensorflow.keras.models import Sequential



######## Read data ########
df = pd.read_csv('../resources/input/full_pair_data.csv')
df.reset_index()

classes = [1, 2, 3, 4]
n_classes  = 4
models_names = []
models_scores = []
random.seed(1024)

######## Get X and Y ########
seqs = df.Seqs.values
tokenizer = Tokenizer(char_level=True)
# fit_on_texts: This method creates the vocabulary index based on word frequency
tokenizer.fit_on_texts(seqs)
# texts_to_sequences: It basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. 
X = tokenizer.texts_to_sequences(seqs)
# pad_sequences --> transforms a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). 
max_length = 1064
X = sequence.pad_sequences(X, maxlen = max_length)
Y = df.Class
#Y = preprocessing.label_binarize(Y, classes=[0, 1, 2, 3])


######## Split train and test ########
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify= Y, random_state=42)

######## Functions ########

# Train and test model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy_value= accuracy_score(y_test, prediction)
    confusion_matrix_model = confusion_matrix(y_test, prediction)
    model_name = str(model.__class__.__name__)
    models_names.append(model_name)
    models_scores.append(accuracy_value)
    plot_confusion_matrix (confusion_matrix_model,classes, model_name)
    #plot_roc_curve(model, model_name)
    print(str(model) + ": Accuracy = " + str(accuracy_value))
    return accuracy_value


# Compute normalized confusion matrix

def plot_confusion_matrix(confusion_matrix, classes, model_name, title = 'Confusion matrix', cmap = plt.cm.Blues):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print(confusion_matrix) 
    #output_text.write(confusion_matrix)
    plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title + ' ' + model_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' 
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./output_images/confusion_matrix_' + model_name + '.png')

def plot_roc_curve(model, model_name):
    Y = df.Class
    Y = preprocessing.label_binarize(Y, classes=[0, 1, 2, 3])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify= Y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

    # Plot of a ROC curve for a specific class
    plt.figure()
    fig, axs = plt.subplots(1, 4, figsize=(15, 6))
    axs = axs.ravel()
    fig.subplots_adjust(hspace = .5, wspace=.01)

    for i in range(n_classes):
        axs[i].plot(false_positive_rate[i], true_positive_rate[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        axs[i].plot([0, 1], [0, 1], 'k--')
        axs[i].title.set_text('Class' + str(i+1))
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
    fig.suptitle(model_name + ' ROC Curve')
    plt.savefig('./output_images/roc_curve_' + model_name + '.png')

######## Create models ########

#-------- Decision Tree model --------#
# Create model
decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=3,
                       min_samples_split=8)

accuracy_decision_tree_model = evaluate_model(decision_tree_model, X_train, X_test, y_train, y_test) 

#-------- Random Forest model --------#
# Create model
random_forest_model = RandomForestClassifier(criterion='entropy', max_depth=20, min_samples_leaf=5,
                       min_samples_split=12, n_estimators=400, random_state=0)
accuracy_random_forest_model = evaluate_model(random_forest_model, X_train, X_test, y_train, y_test) 

#-------- Gradient Boosting model --------#
# Create model
gradient_boosting_model = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features=7,
                           min_samples_leaf=60, min_samples_split=400,
                           n_estimators=300, random_state=10, subsample=0.85)
accuracy_gradient_boosting_model = evaluate_model(gradient_boosting_model, X_train, X_test, y_train, y_test) 

#-------- AdaBoost model --------#
# Create model
adaboost_model = AdaBoostClassifier(learning_rate=0.4, n_estimators=300)
accuracy_adabost_model = evaluate_model(adaboost_model, X_train, X_test, y_train, y_test) 
print('OK')

#-------- K-nearest Neighbours model --------#
# Create model
knn_model = KNeighborsClassifier(leaf_size=1, p=1)
accuracy_knn_model = evaluate_model(knn_model, X_train, X_test, y_train, y_test) 


#-------- CNN + BiLSTM model --------#

def create_CNN_BiLSTM(optimizer='nadam'):
    embedding_dim = 21
    top_classes = 4
    model_CNN_BiLSTM = Sequential()
    model_CNN_BiLSTM.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
    model_CNN_BiLSTM.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='selu'))
    model_CNN_BiLSTM.add(MaxPooling1D(pool_size=2))
    model_CNN_BiLSTM.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='selu'))
    model_CNN_BiLSTM.add(MaxPooling1D(pool_size=2))
    model_CNN_BiLSTM.add(LSTM(16, input_length=128, input_dim=32, return_sequences=True))
    model_CNN_BiLSTM.add(Flatten())
    model_CNN_BiLSTM.add(Dense(1024, activation='selu'))
    model_CNN_BiLSTM.add(Dense(128, activation='selu'))
    model_CNN_BiLSTM.add(Dense(top_classes, activation='softmax'))
    model_CNN_BiLSTM.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #plot_model(model_CNN_BiLSTM, to_file = './output_images/model_CNN_BiLSTM.png', show_shapes = False, show_layer_names = True, rankdir = 'TB', expand_nested = False, dpi = 96)
    return model_CNN_BiLSTM


model_CNN_BiLSTM = create_CNN_BiLSTM('rmsprop')
model_CNN_BiLSTM.fit(X_train, y_train, epochs=5,batch_size=512)
prediction_CNN_BiLSTM = model_CNN_BiLSTM.predict(X_test)
prediction_CNN_BiLSTM = np.argmax(prediction_CNN_BiLSTM, axis=1)
confusion_matrix_CNN_BiLSTM = confusion_matrix(y_test, prediction_CNN_BiLSTM)
model_name_BiLSTM = 'CNN_BiLSTM'
plot_confusion_matrix (confusion_matrix_CNN_BiLSTM, classes, model_name_BiLSTM)
accuracy_value_CNN_BiLSTM= accuracy_score(y_test, prediction_CNN_BiLSTM)
print(str(model_name_BiLSTM) + ": Accuracy = " + str(accuracy_value_CNN_BiLSTM))
model_CNN_BiLSTM.save("model/model_CNN_BiLSTM.h5")

#-------- CNN model --------#

def create_CNN(optimizer='nadam'):
    embedding_dim = 21
    top_classes = 4
    model_CNN = Sequential()
    model_CNN.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
    model_CNN.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='selu'))
    model_CNN.add(MaxPooling1D(pool_size=2))
    model_CNN.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='selu'))
    model_CNN.add(MaxPooling1D(pool_size=2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(top_classes, activation='softmax'))
    model_CNN.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model_CNN

model_CNN = create_CNN()
model_CNN.fit(X_train, y_train, epochs = 5, batch_size = 512)
prediction_CNN = model_CNN.predict(X_test)
accuracy_value_CNN = accuracy_score(y_test, prediction_CNN)
confusion_matrix_CNN = confusion_matrix(y_test, prediction_CNN)
model_name_CNN = 'CNN'
plot_confusion_matrix (confusion_matrix_CNN, classes, model_name_CNN)
print(str(model_name_CNN) + ": Accuracy = " + str(accuracy_value_CNN))
model_CNN.save("model/model_CNN.h5")
