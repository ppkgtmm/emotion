from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, GRU
from keras.models import Sequential
from .utils import get_optimizer, get_loss, get_metr

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('max_len'),)))
    model.add(Embedding(hp.get('num_words'), hp.get('output_dim')))
    for i in range(5,hp.get('log2_filter')+1):
        model.add(Conv1D(2**i, hp.get('kernel_size'), activation=hp.get('activation')))
        model.add(MaxPool1D(hp.get('pool_size')))
    model.add(GRU(hp.get('gru_units')))
    model.add(Dense(6, activation='softmax'))
    model.compile(
        optimizer=get_optimizer(hp.get('optimizer')),
        loss=get_loss(binary=False),
        metrics=get_metr(binary=False)
    )
    model.summary()
    return model