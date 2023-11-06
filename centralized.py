import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from metricas import model_evaluation
from preparas_SEQ import generate_data_semanal
from sklearn.utils.class_weight import compute_class_weight
from redes import bilstm
import tensorflow

EPOCHS=150
BATCH=64
VS=0.15
es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,patience=10,mode='max',restore_best_weights=True)
######################################################################
auc=[]
########################################################################

# 10-fold cross-validation loop
for i in range(10):
    # Define the data for each entity
    X, y = generate_data_semanal('dataset1')
    X2, y2 = generate_data_semanal('dataset2')
    X3, y3 = generate_data_semanal('dataset3')

    # Split the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=i, stratify=y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=i, stratify=y2)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2,random_state=i, stratify=y3)

    # Concatenate X and y of the three datasets
    X = np.concatenate((X_train, X2_train, X3_train), axis=0)
    y = np.concatenate((y_train, y2_train, y3_train), axis=0)

    #Compute class weights
    class_weights = compute_class_weight(class_weight ='balanced',classes =np.unique(y),y=y)
    class_weights = dict(zip(np.unique(y), class_weights))

    # fit the model
    model=None
    model = bilstm()
    history = model.fit(X,y,batch_size=BATCH,epochs=EPOCHS,validation_split=VS,callbacks=[es], shuffle=True, verbose=0
            ,class_weight=class_weights)

    #predict test data
    y_pred=model.predict(X3_test) #change between X_test, X2_test, X3_test to test in the different datasets
    
    # calculate optimal threshold
    fp_r, tp_r, t = roc_curve(y3_test, y_pred)
    th=t[np.argmax(tp_r - fp_r)]
    y_pred = np.where(y_pred>th,1,0)
            
    aucs=model_evaluation(y3_test, y_pred)
            
    # save results
    auc.append(aucs)


# print results
print('*'*60)
print('Resultados')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')

