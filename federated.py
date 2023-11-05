import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras_preprocessing.sequence import pad_sequences
from metricas import model_evaluation
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve
from preparas_SEQ import generate_data_semanal
from utils import *
from redes import bilstm

EPOCHS=150
BATCH=64
VS=0.25
es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1,patience=10,mode='min',restore_best_weights=True)
######################################################################
auc=[]
########################################################################

def batch_data(data_shard, bs=BATCH):
    """
    Make a dataset that shuffles and batches the data.
    
    data_shard: A tuple containing the data and labels. 
    bs: Determines the number of samples that will be included in each batch of the dataset.
    
    Return: 
    A TensorFlow Dataset object that has been shuffled and batched.
    """
   # separa la porción en listas de datos y etiquetas
    data, label = data_shard
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    return dataset.shuffle(len(label)).batch(bs)

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

for rs in range(10):
    # Split the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=rs, stratify=y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=rs, stratify=y2)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2,random_state=rs, stratify=y3)
    
    #split val set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=rs, stratify=y_train)
    X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2,random_state=rs, stratify=y2_train)
    X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train, y3_train, test_size=0.2,random_state=rs, stratify=y3_train)


    # Create clients
    clients_batched = {
        'client1': batch_data((X_train, y_train)),
        'client2': batch_data((X2_train, y2_train)),
        'client3': batch_data((X3_train, y3_train))
    }

    #Compute class weights
    class_weights1 = compute_class_weight(class_weight ='balanced',classes =np.unique(y_train),y=y_train)
    class_weights1 = dict(zip(np.unique(y_train), class_weights1))

    class_weights2 = compute_class_weight(class_weight ='balanced',classes =np.unique(y2_train),y=y2_train)
    class_weights2 = dict(zip(np.unique(y2_train), class_weights2))

    class_weights3 = compute_class_weight(class_weight ='balanced',classes =np.unique(y3_train),y=y3_train)
    class_weights3 = dict(zip(np.unique(y3_train), class_weights3))

    class_weights = {
        'client1': class_weights1,  # Pesos de clase para el cliente 1
        'client2': class_weights2,   # Pesos de clase para el cliente 2
        'client3': class_weights3,   # Pesos de clase para el cliente 2
        # Agrega más pesos de clase para otros clientes si es necesario
    }

    # Define the global model
    global_model = bilstm()
    comms_round = 1

    # Training loop
    for comm_round in range(comms_round):
        global_weights = global_model.get_weights()
        scaled_local_weight_list = []

        for client, client_data in clients_batched.items():
            local_model = bilstm()
            
            local_model.set_weights(global_weights)
            
            class_weights_client = class_weights.get(client, None)  # obtain weights to the client

            if client=='client1':
                local_model.fit(client_data, batch_size=BATCH ,epochs=EPOCHS, validation_data=(X_val, y_val),callbacks=[es], verbose=0, class_weight=class_weights_client)
            elif client=='client2':
                local_model.fit(client_data, batch_size=BATCH ,epochs=EPOCHS, validation_data=(X2_val, y2_val),callbacks=[es], verbose=0, class_weight=class_weights_client)
            elif client=='client3':
                local_model.fit(client_data, batch_size=BATCH ,epochs=EPOCHS, validation_data=(X3_val, y3_val),callbacks=[es], verbose=0, class_weight=class_weights_client)
            
            scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            
            #clear session to free memory after each communication round
            K.clear_session()

        average_weights = sum_scaled_weights(scaled_local_weight_list)
        #update global model 
        global_model.set_weights(average_weights)
        K.clear_session()
    
    #predict final model
    y_pred=global_model.predict(X3_test) #change between (X_test,y_test), (X2_test,y2_test) and (X3_test,y3_test) to test the different datasets
    
    # calculate optimal threshold
    fp_r, tp_r, t = roc_curve(y3_test, y_pred)
    th=t[np.argmax(tp_r - fp_r)]
    y_pred = np.where(y_pred>th,1,0)
        
    aucs=model_evaluation(y3_test, y_pred)
        
    # Guardar resultados
    auc.append(aucs)


# Imprime los resultados
print('*'*60)
print('Results')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')

