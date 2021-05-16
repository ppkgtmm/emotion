from keras.layers import Input, Dense, Embedding, GRU, Bidirectional
from keras.models import Sequential
from .utils import get_optimizer, get_loss, get_metr

def build_bigru(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    model.add(Embedding(hp.get('num_words'), hp.get('output_dim')))
    model.add(Bidirectional(GRU(hp.get('gru_units'))))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model

def build_stacked_bigru(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    model.add(Embedding(hp.get('num_words'), hp.get('output_dim')))
    for i in range(hp.get('n_layers')):
        model.add(Bidirectional(GRU(hp.get('gru_units'), return_sequences= (i+1 < hp.get('n_layers')))))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model

def build_dstacked_bigru(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    model.add(Embedding(hp.get('num_words'), hp.get('output_dim')))
    model.add(Bidirectional(GRU(hp.get('gru_units'), return_sequences= True, dropout=hp.get('drop_rate'))))
    model.add(Bidirectional(GRU(hp.get('gru_units'))))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model