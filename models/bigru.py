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