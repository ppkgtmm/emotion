from .utils import get_optimizer, get_loss, get_metr
from keras.layers import Input, Dense
from keras.models import Sequential

def build_mlp(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    model.add(Dense(hp.get('units'), activation=hp.get('activation')))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model

def build_mlp_oe(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    for i in range(hp.get('log_units'),2,-1):
        model.add(Dense(2**i, activation=hp.get('activation')))
    model.add(Dense(2**2, activation=hp.get('activation')))
    for i in range(2+1,hp.get('log_units')+1,1):
        model.add(Dense(2**i, activation=hp.get('activation')))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model

def build_dmlp(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    for i in range(2,hp.get('log_units'),1):
        model.add(Dense(2**i, activation=hp.get('activation')))
    model.add(Dense(2 ** hp.get('log_units'), activation=hp.get('activation')))
    for i in range(hp.get('log_units')-1,1,-1):
        model.add(Dense(2**i, activation=hp.get('activation')))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model