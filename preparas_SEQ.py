import pandas as pd
import numpy as np
import tensorflow as tf


def convert_to_tfrecord(sequences, labels, tfrecord_file):
    """
    The function `convert_to_tfrecord` converts sequences and labels into a TFRecord format and writes
    them to a TFRecord file.
    
    sequences: A list of sequences, where each sequence is a numpy array of floats
    labels: The "labels" parameter is a list or array containing the labels corresponding to each
    sequence. Each label represents the class or category that the sequence belongs to
    tfrecord_file: The tfrecord_file parameter is the file path where the TFRecord file will be
    saved. It should have the extension ".tfrecord"
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for sequence, label in zip(sequences, labels):
            # Convertir la secuencia y la etiqueta en formatos compatibles con TFRecord
            sequence_feature = tf.train.Feature(float_list=tf.train.FloatList(value=sequence.flatten()))
            label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            
            # Crear un ejemplo de TFRecord
            example = tf.train.Example(features=tf.train.Features(feature={
                'sequence': sequence_feature,
                'label': label_feature
            }))
            
            # Escribir el ejemplo en el archivo TFRecord
            writer.write(example.SerializeToString())

def generate_data_semanal(file):    
    """
    The function `generate_data_semanal` reads a CSV file, removes unnecessary columns, groups the data
    by user_enrolment_id, extracts the necessary variables and labels for each user, and returns the
    extracted variables and labels as numpy arrays.
    
    file: The `file` parameter is the name or path of the CSV file that contains the data you
    want to process
    Return:
    The function `generate_data_semanal` returns two arrays: `X_local` and `y_local`. `X_local`
    contains the extracted variables from each user, while `y_local` contains the corresponding labels.
    """
    # Leer el archivo CSV
    df = pd.read_csv(f'{file}',header=0, index_col=0)
    
    # Se inicializan las listas
    X_local = []
    y_local=[]
    
    # Agrupar los datos por user_enrolment_id y procesar cada grupo.
    for _, grupo in df.groupby('user_enrolment_id'):
        # Extraer las variables necesarias de cada usuario 
        x = grupo.iloc[:, 3:-1]
        # Extraer la etiqueta correspondiente
        y=  grupo.iloc[0, -1]
        #Agregamos los valores a las listas
        X_local.append(x.values)
        y_local.append(y)
    
    return np.array(X_local,dtype=object), np.array(y_local)


