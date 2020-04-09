import tensorflow as tf
import numpy as np


class Generator:

    def __init__(self, rnn_units, embedding_dim, seq_legnth, vocab, batch_size):

        super(Generator, self).__init__()
        self.rnn_units = rnn_units
        self.embedding_dim = embedding_dim
        self.seq_length = seq_legnth
        self.vocab = vocab
        self.batch_size = batch_size
        self.vocab_size = len(vocab)

        self.char2idx = {i: u for i, u in enumerate(vocab)}
        self.idx2char = np.array(vocab)

        self.optimizer = None
        self.model = None
        self.loss_object = None

    def build_model(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(100, input_shape=(self.seq_length, self.vocab_size)))
        model.add(tf.keras.layers.Dense(self.seq_length, activation='sigmoid'))

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        # model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.model = model

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
