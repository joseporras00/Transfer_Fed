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

def weight_scalling_factor(clients_trn_data, client_name):
    """
    The function calculates the weight scaling factor for a specific client based on the total number of
    data points held by that client and the total number of data points across all clients.
    
    clients_trn_data: dictionary where the keys are client names and the values are the training data for each client. 
    client_name: name of the client for which you want to calculate the weight scaling factor
    
    Return: the weight scaling factor, which is the ratio of the total number of data points held by a
    specific client to the total number of data points across all clients.
    """
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    """
    Scale the weights of a model.
    
    weight: list of numbers representing the weights of a model
    scalar: number that will be used to scale the model's weights
    
    Return:
    a list of scaled weights.
    """
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    """
    The function calculates the average gradient across all client gradients and returns the sum of the
    listed scaled weights.
    
    scaled_weight_list: list of lists. Each list represents the scaled weights for a particular layer. 
    
    Return: 
    a list of average gradients.
    """
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

# Define the data for each entity
X, y = generate_data_semanal('dataset1')
X2, y2 = generate_data_semanal('dataset2')
X3, y3 = generate_data_semanal('dataset3')

# 10-fold cross-validation loop
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

    #C ompute class weights
    class_weights1 = compute_class_weight(class_weight ='balanced',classes =np.unique(y_train),y=y_train)
    class_weights1 = dict(zip(np.unique(y_train), class_weights1))

    class_weights2 = compute_class_weight(class_weight ='balanced',classes =np.unique(y2_train),y=y2_train)
    class_weights2 = dict(zip(np.unique(y2_train), class_weights2))

    class_weights3 = compute_class_weight(class_weight ='balanced',classes =np.unique(y3_train),y=y3_train)
    class_weights3 = dict(zip(np.unique(y3_train), class_weights3))

   # The `class_weights` dictionary used to assign different weights to each class in the training
   # data for each client. 
    class_weights = {
        'client1': class_weights1, 
        'client2': class_weights2,  
        'client3': class_weights3, 
    }

    # Define the global model
    global_model = bilstm()
    
    # Define rounds, 1 because we use all the dats
    comms_round = 1

    # Federated Learning Training loop
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
    
    # prediction for final model change between (X_test,y_test), (X2_test,y2_test) and (X3_test,y3_test) 
    # to test the different datasets
    y_pred=global_model.predict(X3_test) 
    
    # calculate optimal threshold
    fp_r, tp_r, t = roc_curve(y3_test, y_pred)
    th=t[np.argmax(tp_r - fp_r)]
    y_pred = np.where(y_pred>th,1,0)
        
    aucs=model_evaluation(y3_test, y_pred)
        
    # save results
    auc.append(aucs)


# print results
print('*'*60)
print('Results')
print(f'Mean AUC: {sum(auc) / len(auc)}')
print(f'STD AUC: {np.std(auc)}')

