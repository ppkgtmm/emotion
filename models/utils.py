from keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision,\
Recall, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.optimizers import Nadam, Adam
from typing import List
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, \
confusion_matrix
import matplotlib.pyplot as plt

seed = 123456
OH_encoder = None
columns = ['text', 'target']
data_path = '/content/drive/MyDrive/emotion/data/emotion_nlp.csv'
num_words = 10000
max_len = 250
optimizers = [
           'adam',
           'nadam',
           'amsgrad'   
]

def read_data(data_path=data_path, cols=columns):
    if cols:
        return pd.read_csv(data_path)[cols]

    return pd.read_csv(data_path)

def split_data(data, test_size=0.15, stratify='target'):
    return train_test_split(
        data,
        test_size=test_size,
        stratify=data[stratify],
        random_state=seed
    )

def read_and_split(data_path=data_path, test_size=0.15, cols=columns):
    data = read_data(data_path, cols)
    return split_data(data, test_size, cols[-1])

def OH_fit_transform(col):
    OH_fit(col)
    return OH_transform(col)

def OH_fit(col):
    global OH_encoder
    OH_encoder = OneHotEncoder(sparse=False)
    OH_encoder.fit(col.values.reshape(-1, 1))

def OH_transform(col):
    return OH_encoder.transform(col.values.reshape(-1, 1))
     
def get_optimizer(key):
    map = {
      'adam': Adam(),
      'nadam': Nadam(),
      'amsgrad': Adam(amsgrad=True),
    }
    return map.get(key)

def dump(obj, path):
    return pickle.dump(obj, open(path, 'wb'))

def load(path):
    return pickle.load(open(path, 'rb'))

def get_callbacks(file_path, monitor='val_accuracy', patience=5):
    return [
          EarlyStopping(monitor=monitor, patience=patience, \
                        restore_best_weights=True),
          ModelCheckpoint(file_path, monitor=monitor, verbose=1, \
                          save_best_only=True,  mode='max')
    ]
def get_sequences(tokenizer, texts, max_len=max_len):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

def get_binary_loss():
  return BinaryCrossentropy()

def get_multi_loss():
  return CategoricalCrossentropy()
  
def get_loss(binary=True):
  if binary:
    return get_binary_loss()
  return get_multi_loss()

def get_binary_metr():
  return [
          'accuracy',
          Precision(),
          Recall(),
          TruePositives(),
          TrueNegatives(),
          FalsePositives(),
          FalseNegatives()
  ]

def get_multi_metr():
  return [
          'accuracy',
          Precision(),
          Recall()
  ]
  
def get_metr(binary=True):
  if binary:
    return get_binary_metr()
  return get_multi_metr()

def get_model_from_config(base_line, optimizer, binary=True):
    model = Sequential().from_config(base_line.get_config())
    model.compile(
        optimizer=get_optimizer(optimizer),
        loss=get_loss(binary),
        metrics=get_metr(binary)
    )
    return model


def model_evaluate(model, X_test, y_test, batch_size=32, normalize="true"):
    plt.rc('font', **{'size': 13})
    # predict class with test set
    y_pred_test =  model.predict(X_test, batch_size=batch_size, verbose=1)
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred_test,axis=1))*100))
    
    #classification report
    print('\n')
    assert OH_encoder is not None
    inverse_y_test = OH_encoder.inverse_transform(y_test).reshape(-1, 1)
    inverse_y_pred = OH_encoder.inverse_transform(y_pred_test).reshape(-1, 1)
    classes = np.unique(inverse_y_test)
    print(classification_report(inverse_y_test, inverse_y_pred))
    #confusion matrix
    confmat = confusion_matrix(inverse_y_test, inverse_y_pred, normalize=normalize, labels=classes)
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(confmat, cmap=plt.cm.Purples, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=round(confmat[i, j],3), va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(np.arange(len(classes)), classes, rotation=45, fontsize=14)
    plt.yticks(np.arange(len(classes)), classes, fontsize=14)
    plt.tight_layout()