import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from metricas import model_evaluation
from preparas_SEQ import generate_data_semanal
from sklearn.utils.class_weight import compute_class_weight
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import tensorflow

EPOCHS=150
BATCH=64
VS=0.25
es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,patience=10,mode='max',restore_best_weights=True)
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

# split data into training and val set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=42, stratify=y2)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2,random_state=42, stratify=y3)

#Compute class weights
class_weights1 = compute_class_weight(class_weight ='balanced',classes =np.unique(y_train),y=y_train)
class_weights1 = dict(zip(np.unique(y_train), class_weights1))

class_weights2 = compute_class_weight(class_weight ='balanced',classes =np.unique(y2_train),y=y2_train)
class_weights2 = dict(zip(np.unique(y2_train), class_weights2))

class_weights3 = compute_class_weight(class_weight ='balanced',classes =np.unique(y3_train),y=y3_train)
class_weights3 = dict(zip(np.unique(y3_train), class_weights3))

def evaluate_model(X, y, class_weights, aucs, n_folds=10, X_val=None, y_val=None):
    """
    The function performs k-fold cross-validation on a given dataset using a specified
    model, and returns the training histories and AUC scores for each fold.
    
    X: The input features for the model
    y:The target variable or the labels of the data. 
    class_weights: Dictionary that assigns weights to each class in the target variable. 
    aucs: List that stores the calculated AUC (Area Under the Curve) values for each fold of the cross-validation. 
    n_folds: The number of folds for cross-validation. Defaults to 10 (optional)
    X_val: Validation data, which is used to make decisions on when to stop training (early stopping) 
    y_val: The validation set labels. 
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
        
        # fit model        
        #model=None
        model=load_model('model1.h5') #Change between model1, model2 and model3
        
        # retrain to tune the model
        history = model.fit(X_train,y_train,batch_size=BATCH,epochs=EPOCHS,validation_data=(X_val,y_val),callbacks=[es], verbose=0
        ,class_weight=class_weights) 
        
        # predict test data
        y_pred=model.predict(X_test)
        
        # calculate optimal threshold
        fp_r, tp_r, t = roc_curve(y_test, y_pred)
        th=t[np.argmax(tp_r - fp_r)]
        y_pred = np.where(y_pred>th,1,0)
        
        auc=model_evaluation(y_test, y_pred)
        
        # save resuls
        aucs.append(auc)
        
    return histories, aucs


histories, auc = evaluate_model(X_train, y_train, class_weights1, auc, X_val=X_test, y_val=y_test)
# print results in dataset1
print('*'*60)
print('Results Dataset1')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')

######################################################################
auc=[]
########################################################################

histories, auc = evaluate_model(X2_train, y2_train, class_weights2, auc, X_val=X2_test, y_val=y2_test)
# print results in dataset2
print('*'*60)
print('Results Dataset2')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')

######################################################################
auc=[]
########################################################################

histories, auc = evaluate_model(X3_train, y3_train, class_weights3, auc, X_val=X3_test, y_val=y3_test)
# print results in dataset3
print('*'*60)
print('Resuls Dataset3')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')



