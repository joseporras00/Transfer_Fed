import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from metricas import model_evaluation
from preparas_SEQ import generate_data_semanal
from sklearn.utils.class_weight import compute_class_weight
from metricas import model_evaluation
from redes import *
from sklearn.model_selection import StratifiedKFold
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow
import time

EPOCHS=150
BATCH=64
VS=0.25
es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,patience=10,mode='max',restore_best_weights=True)
######################################################################
auc=[]
########################################################################

#  define the data for the entity (dataset1), (dataset2), (dataset3)
# change to test other dataset
X, y = generate_data_semanal('dataset1')

# padding the sequences to have the same length
max_seq=max(len(elem) for elem in X)
special_value=-10.0
X = pad_sequences(X, maxlen=max_seq,dtype='float', padding='post', truncating='post', value=special_value)

# split train and val sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)

# compute class weights
class_weights = compute_class_weight(class_weight ='balanced',classes =np.unique(y_train),y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# load models
model1=load_model('model1.h5')
model2=load_model('model2.h5')
model3=load_model('model3.h5')

# make predictions to include to the train dataset
outputs1 = model1.predict(X_train)
outputs2 = model2.predict(X_train)
outputs3 = model3.predict(X_train)

# concatenate to train matrix data
new_data_matrix = []
for i in range(len(X_train)):
    sample = X_train[i]
    num_sequences = sample.shape[0]
    output1 = np.tile(outputs1[i], (num_sequences, 1))
    output2 = np.tile(outputs2[i], (num_sequences, 1))
    output3 = np.tile(outputs3[i], (num_sequences, 1))
    
    # Concatenate
    combined = np.concatenate((sample, output1, output2, output3), axis=1)
    
    new_data_matrix.append(combined)

# convert to array
X_train = np.array(new_data_matrix)

# make predictions to include to the test dataset
outputs1 = model1.predict(X_test)
outputs2 = model2.predict(X_test)
outputs3 = model3.predict(X_test)

# concatenate to test matrix data
new_data_matrix = []
for i in range(len(X_test)):
    sample = X_test[i]
    num_sequences = sample.shape[0]
    output1 = np.tile(outputs1[i], (num_sequences, 1))
    output2 = np.tile(outputs2[i], (num_sequences, 1))
    output3 = np.tile(outputs3[i], (num_sequences, 1))
    
    # Concatenate
    combined = np.concatenate((sample, output1, output2, output3), axis=1)
    
    new_data_matrix.append(combined)
    
# convert to array
X_test= np.array(new_data_matrix)

def evaluate_model(X, y, X_val, y_val, aucs, n_folds=10):
    """
    The function performs k-fold cross-validation, trains a model on each fold, makes
    predictions on a test set, and evaluates the model's.
    
    X: The input features for training the model
    y: The target variable for the training data.
    X_test: X_test is the input data for testing the model.
    y_test: The parameter `y_test` is the true labels for the test set.
    aucs: list that is used to store the AUC (Area Under the Curve) values for each fold of the cross-validation
    n_folds: number of folds to be used. In this case, it is set to 10 (optional).
    """
    kfold = StratifiedKFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X, y):  
        print('KFOLD------------------------------------')      
        # select rows for train and test
        X_train, y_train, X_test, y_test =  X[train_ix], y[train_ix], X[test_ix], y[test_ix]  
        unique, counts = np.unique(y_train, return_counts=True)
        dic=dict(zip(unique, counts))
        print(dic)
        
        # fit model   
        combined_model=bilstm_stacking()
        combined_model.fit(X_train,y_train,batch_size=BATCH,epochs=EPOCHS,validation_data=(X_val,y_val),callbacks=[es], verbose=0
         #The class weights go here
        ,class_weight=class_weights)
        
        # make predictions        
        y_pred=combined_model.predict(X_test)
        y_pred=np.round(y_pred)
        
        auc=model_evaluation(y_test, y_pred)
        
        # save resuls
        aucs.append(auc)
        
evaluate_model(X_train, y_train, X_test, y_test, auc)

# print results
print('*'*60)
print('Results:')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')