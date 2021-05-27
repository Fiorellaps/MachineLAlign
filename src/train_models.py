import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.datasets import make_moons
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding, Dropout, Bidirectional, GRU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold 

######## Read data ########
df = pd.read_csv('../resources/input/full_pair_data.csv')
df.reset_index()
#df.shape
#df.head()
#df.info()
classes = [1, 2, 3, 4]
models_names = []
models_scores = []
random.seed(1024)
stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
for x, y in stratified_split.split(df, df['Class']):
    stratified_df = df.iloc[y]

sns.countplot(stratified_df['Class'])
plt.savefig("./output_images/class_distribution.png")

plt.figure()
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(stratified_df['Class'])
f.add_subplot(1,2,2)
sns.boxplot(stratified_df['Class'])
plt.savefig("./output_images/class_distplot_boxplot.png")
#output_text.write(str(stratified_df.shape[0]) + " instances")
print(str(stratified_df.shape[0]) + " instances")



######## Get X and Y ########
seqs = stratified_df.Seqs.values
tokenizer = Tokenizer(char_level=True)
# fit_on_texts: This method creates the vocabulary index based on word frequency
tokenizer.fit_on_texts(seqs)
# texts_to_sequences: It basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. 
X = tokenizer.texts_to_sequences(seqs)
# pad_sequences --> transforms a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). 
max_length = 1064
X = sequence.pad_sequences(X, maxlen = max_length)
Y = stratified_df.Class

######## Split train and test ########
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify= Y, random_state=42)

######## Functions ########

# Train and test model
def evaluate_model(model, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy_value= accuracy_score(y_test, prediction)
    confusion_matrix_model = confusion_matrix(y_test, prediction)
    model_name = str(model.__class__.__name__)
    models_names.append(model_name)
    models_scores.append(accuracy_value)
    plot_confusion_matrix (confusion_matrix_model,classes, model_name)
    #output_text.write(str(model) + ": Accuracy = " + str(accuracy_value))
    print(str(model) + ": Accuracy = " + str(accuracy_value))
    return accuracy_value


# Hyperparemeters tuning
def hiperparametrization_grid_search(model, prameters_grid, X_train , X_test , y_train , y_test ):
    model_grid_search = GridSearchCV(estimator = model, param_grid = prameters_grid, cv=10, n_jobs = -1)
    model_grid_search.fit(X_train, y_train)
    model_grid_search.best_params_
    best_model = model_grid_search.best_estimator_
    prediction = best_model.predict(X_test)
    accuracy_value= accuracy_score(y_test, prediction)
    confusion_matrix_model = confusion_matrix(y_test, prediction)
    model_name = str(model.__class__.__name__)
    models_names.append(model_name)
    models_scores.append(accuracy_value)
    plot_confusion_matrix (confusion_matrix_model,classes, model_name)
    #output_text.write(str(best_model) + ": Accuracy = " + str(accuracy_value))
    print(str(best_model) + ": Accuracy = " + str(accuracy_value))
    print(model_grid_search.best_params_)
    #joblib.dump(best_model, './model/model_' + model_name + '.pkl') 
    return best_model

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

######## Create models ########
'''
#-------- Decision Tree model --------#
# Create model
decision_tree_model = DecisionTreeClassifier(min_samples_leaf= 3)
#decision_tree_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, min_samples_leaf= 3)

#accuracy_decision_tree_model = evaluate_model(decision_tree_model, X_train, X_test, y_train, y_test) 
#Define parameters
parameter_grid_decision_tree = {
    "max_depth": range(1 ,10, 2),
    "min_samples_split": range(2, 10, 2),
    "criterion": ["gini", "entropy"]
}

# Grid Search Tuning
decision_tree_best_model = hiperparametrization_grid_search(decision_tree_model, parameter_grid_decision_tree, X_train, X_test, y_train, y_test)

#-------- Random Forest model --------#
# Create model
random_forest_model = RandomForestClassifier(n_estimators = 100, max_features = 'auto', random_state = 0, min_samples_leaf = 5, min_samples_split = 12)
#random_forest_model = RandomForestClassifier(n_estimators = 300, max_depth = 30, criterion = 'entropy', max_features = 'auto', random_state = 0, min_samples_leaf = 5, min_samples_split = 12)

#accuracy_random_forest_model = evaluate_model(random_forest_model, X_train, X_test, y_train, y_test) 
#Define parameters
parameter_grid_random_forest = {
    "criterion": ['gini', 'entropy'],
    "n_estimators": [200, 300, 400],
    "max_depth": range(20, 50 ,10), 
}
# Grid Search Tuning
random_forest_best_model = hiperparametrization_grid_search(random_forest_model, parameter_grid_random_forest, X_train, X_test, y_train, y_test)

#-------- Gradient Boosting model --------#
# Create model
gradient_boosting_model = GradientBoostingClassifier(min_samples_leaf = 60, subsample = 0.85, random_state = 10, max_features = 7)
#gradient_boosting_model = GradientBoostingClassifiern_estimators(n_estimators = 300, learning_rate = 0.2, max_depth = 7, min_samples_split = , 600, min_samples_leaf = 60, subsample = 0.85, random_state = 10, max_features = 7)
#accuracy_gradient_boosting_model = evaluate_model(gradient_boosting_model, X_train, X_test, y_train, y_test) 
#Define parameters
parameter_grid_gradient_boosting = {
    "n_estimators": range(100, 400, 100),
    "max_depth": range(2, 9, 2), 
    "min_samples_split": range(400, 700, 100),
    "learning_rate": [0.1, 0.2]
}

# Grid Search Tuning
gradient_boosting_best_model = hiperparametrization_grid_search(gradient_boosting_model, parameter_grid_gradient_boosting, X_train, X_test, y_train, y_test)

#-------- AdaBoost model --------#
# Create model
adaboost_model = AdaBoostClassifier(n_estimators = 300, learning_rate = 0.3)
#adaboost_model = AdaBoostClassifier(n_estimators = 100)

#accuracy_adabost_model = evaluate_model(adaboost_model, X_train, X_test, y_train, y_test) 
#Define parameters
parameter_grid_adaboost = {
    "n_estimators": range(100, 400, 100),
    "learning_rate": [0.2, 0.3, 0.4]
}
# Grid Search Tuning
adaboost_best_model = hiperparametrization_grid_search(adaboost_model, parameter_grid_adaboost, X_train, X_test, y_train, y_test)

#-------- K-nearest Neighbors model --------#
# Create model
knn_model = KNeighborsClassifier(n_neighbors = 7)
#knn_model = KNeighborsClassifier(n_neighbors = 3 , leaf_size = 30, p = 2)
#evaluate_model(knn_model, X_train, X_test, y_train, y_test) 
#Define parameters
parameter_grid_knn = {
	"leaf_size":range(1, 50, 10),
	"n_neighbors": range(2, 16, 3),
	"p":[1, 2]
}
# Grid Search Tuning
knn_best_model = hiperparametrization_grid_search(knn_model, parameter_grid_knn, X_train, X_test, y_train, y_test)


#-------- CNN + BiLSTM model --------#

# create Sequential model 
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
    model_CNN_BiLSTM.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    #plot_model(model_CNN_BiLSTM, to_file = './output_images/model_CNN_BiLSTM.png', show_shapes = False,
    #show_layer_names = True, rankdir = 'TB', expand_nested = False, dpi = 96)
    return model_CNN_BiLSTM
# Plot model


# Train model
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#model_CNN_BiLSTM.fit(X_train, y_train, epochs = 10, batch_size = 512, callbacks = callbacks)
model_CNN_BiLSTM = KerasClassifier(build_fn = create_CNN_BiLSTM, verbose = 0)
parameter_grid_CNN_BiLSTM = {
	"epochs" : [5, 10, 15],
    "batch_size" : [512, 1024],
    "optimizer" : ['rmsprop', 'nadam']
}

CNN_BiLSTM_best_model = hiperparametrization_grid_search(model_CNN_BiLSTM, parameter_grid_CNN_BiLSTM, X_train, X_test, y_train, y_test) 


'''
#-------- CNN + model --------#
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
model_CNN = KerasClassifier(build_fn = create_CNN, verbose = 0)
parameter_grid_CNN = {
	"epochs" : [5, 10, 15],
    "batch_size" : [512, 1024],
    "optimizer" : ['adadelta', 'nadam']
}
#CNN_best_model = hiperparametrization_grid_search(model_CNN, parameter_grid_CNN, X_train, X_test, y_train, y_test) 

model_CNN.fit(X_train, y_train, epochs = 5, batch_size = 512)
prediction = model_CNN.predict(X_test)
accuracy_value= accuracy_score(y_test, prediction)
confusion_matrix_model = confusion_matrix(y_test, prediction)
#model_name = str(model_CNN.__class__.__name__)
#models_names.append(model_name)
models_scores.append(accuracy_value)
plot_confusion_matrix (confusion_matrix_model,classes, 'CNN')
#output_text.write(str(model) + ": Accuracy = " + str(accuracy_value))
print(str(model_CNN) + ": Accuracy = " + str(accuracy_value))
