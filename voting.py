import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from metricas import model_evaluation
from preparas_SEQ import generate_data_semanal
from sklearn.utils.class_weight import compute_class_weight
from metricas import model_evaluation
from redes import *
from sklearn.model_selection import StratifiedKFold
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow

EPOCHS=150
BATCH=64
VS=0.25
es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_cohen_kappa', verbose=1,patience=5,mode='max',restore_best_weights=True)
######################################################################
auc=[]
########################################################################

# Define the data for each entity
X, y = generate_data_semanal('dataset1')
X2, y2 = generate_data_semanal('dataset2')
X3, y3 = generate_data_semanal('dataset3')

# padding the sequences to have the same length
max_seq=max(len(elem) for elem in X)
special_value=-10.0
X = pad_sequences(X, maxlen=max_seq,dtype='float', padding='post', truncating='post', value=special_value)

max_seq=max(len(elem) for elem in X2)
special_value=-10.0
X2 = pad_sequences(X2, maxlen=max_seq,dtype='float', padding='post', truncating='post', value=special_value)

max_seq=max(len(elem) for elem in X3)
special_value=-10.0
X3 = pad_sequences(X3, maxlen=max_seq,dtype='float', padding='post', truncating='post', value=special_value)

def evaluate_model(X, y, aucs, n_folds=10, X_val=None, y_val=None):
    """
    The function performs k-fold cross-validation on a given dataset using soft voting
    and returns the evaluation histories and AUC scores.
    
    X: The input features for the model
    y: The target variable or labels for the dataset
    aucs: list that stores the AUC (Area Under the Curve) values for each fold of the cross-validation.
    n_folds: The number of folds for cross-validation. The default value is 10 (optional).
    X_val: The validation set features
    y_val: The target variable for the validation set.
    """
    histories = list()
    kfold = StratifiedKFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X, y):  
        print('KFOLD------------------------------------')      
        # select rows for train and test
        X_train, y_train, X_test, y_test = X[train_ix], y[train_ix], X[test_ix], y[test_ix]  
        unique, counts = np.unique(y_train, return_counts=True)
        dic=dict(zip(unique, counts))
        print(dic)
        
        # make the preds of the model, change in you are testing for other datasets to test in a zero-shot scenario
        #model1=load_model('model1.h5')
        model2=load_model('model2.h5')
        model3=load_model('model3.h5')
        
        models=[model2,model3]
        y_pred=None
        for model in models:
                if model==models[0]:
                        y_pred=model.predict(X_test)
                else:
                        y_pred=y_pred+model.predict(X_test)

        y_pred=y_pred/len(models)
        y_pred=np.round(y_pred)
        
        auc=model_evaluation(y_test, y_pred)
        
        # Guardar resultados
        aucs.append(auc)
        
    return histories, aucs


histories, auc = evaluate_model(X, y, auc) # change between (X,y), (X2,y2) and (X3,y3)

# print results
print('*'*60)
print('Results')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')





