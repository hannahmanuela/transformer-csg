import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


# TODO slightly sketched about 1010101 of labels in an-abn-bn


START_TOKEN = "<START>"
STOP_TOKEN = '<STOP>'
# TODO check this value
BATCH_SIZE = 64
TRAIN = True


"""_______PREPARING DATA________"""

"""
vocab maps a,b to 1,2 (see vocab.txt file) -- 
"""
vocab = {}
with open('vocab.txt', 'r') as f:
    for i, line in enumerate(f, start=1):
        word = line.split()[0]
        vocab[word] = i


"""
+1 because vocab starting index is 1, 0 is for padding
"""
VOCAB_SIZE = len(vocab)+1


# go from binary to ab
def decode_a_line(line):
    """
    :param line: (encoded) string of 1s and 2s
    :return: the same line, decoded into as and bs according to vocab
    """
    vocab_r = dict([(value, key) for key, value in vocab.items()])
    return ' '.join(''.join([vocab_r[i] for i in line]).split('@@'))


# make line binary
def encode_a_line(line):
    """
    :param line: a (raw) string of as and bs
    :return: the same line, encoded via vocab
    """
    return [vocab[w] for w in line.split()]


"""
extracts raw data and labels from the an-abn-bn text file
"""
string_data = []
label = []
with open('an-abn-bn-data.txt', 'r') as f:

    # TODO why are we adding vocab size?
    for line in f:
        tmp = line.split()
        string_data.append([VOCAB_SIZE] + [vocab[w] for w in tmp[0]] + [VOCAB_SIZE+1])
        label.append(int(tmp[1]))


def pad_data(data):
    """
    :param data: an array of data
    :return: the same data, all lines the same length (padded with 0s at end)
    """
    # pad with 0 at the end
    line_len = max([len(line) for line in data])
    for line in data:
        line.extend([0] * (line_len - len(line)))
    return data


'''
actually pad data
initialize data sets -- 90% train
'''
string_data = pad_data(string_data)

TRAIN_SIZE = len(string_data) * 1 // 15
print(TRAIN_SIZE)

training_data = string_data[:TRAIN_SIZE]
training_label = label[:TRAIN_SIZE]
testing_data = string_data[TRAIN_SIZE:]
testing_label = label[TRAIN_SIZE:]

MAX_LENGTH = max([len(line) for line in string_data])


train_dataset = list(zip(training_data, training_label))


"""_______ACTUAL TRANSFORMER - PROCESSING DATA________"""


def get_angles(pos, i, d_model):
    """
    FROM TF
    :param pos: position values for each data point
    :param i: dimension of model currently in???
    :param d_model: dimension of model - size of layers within the model
    :return: angle for positional encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# copied from TF
def positional_encoding(position, d_model):
    """
    FROM TF
    :param position: how far to look ahead - T of sin/cos
    :param d_model: dimension of model - size of layers within the model
    :return: matrix of positional encoding values (with which to multiply???)
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.keras.backend.cast(pos_encoding, dtype=tf.float32)


# TODO is 50 really necessary? -- only 36 char length of inputs
pos_encoding = positional_encoding(50, 512)


def create_padding_mask(seq):
    """
    FROM TF
    :param seq: input values
    :return: indicates where pad value 0 is present:
        it outputs a 1 at those locations, and a 0 otherwise
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    FROM TF
    :param size: how big the mask matrix is (size x size)
    :return: a matrix with zeros where future should be masked, 1 otherwise
    """
    # gives lower triangular part of matrix
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# copied from TF
def scaled_dot_product_attention(q, k, v, mask):
    """
    FROM TF
    :param q: query [shape == (..., seq_len_q, depth)]
    :param k: key [shape == (..., seq_len_k, depth)]
    :param v: value [shape == (..., seq_len_v, depth_v)]
    :param mask: Float tensor [shape broadcastable to (..., seq_len_q, seq_len_k)] Default: None.

    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    :return: output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.keras.backend.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# TODO use to look at values intermittently -- in comb w/ grapher @ end?
def print_out(q, k, v):
    """
    FROM TF
    :param q: query
    :param k: key
    :param v: values
    :return: current output and attention weights are printed to the console
    """
    temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        """
        FROM TF
        :param d_model: dimension of the model (dim of layers)
        :param num_heads: number of heads that attention is done in (w/ reduced
        dimensionality of d)model/num_heads, thus d_model mod num_heads == 0)
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        discriminator_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        :param x: matrix (such as queries)
        :param batch_size: ???
        :return: x split into batch_size and num_heads
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        :param v: value
        :param k: key
        :param q: query
        :param mask: mask
        :return: output and attention weights
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    FROM TF
    :param d_model: dimension of model
    :param dff: dimension of inner layer (dimension feed forward)
    :return: a feed forward network of inputted dimensions
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        FROM TF
        :param d_model: dimension of model
        :param num_heads: number of heads fro multi-head attention
        :param dff: dimension of feed forward network
        :param rate: dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        :param x: inputs
        :param training: if in training or not
        :param mask: padding mask
        :return: output of encoder layer
        """
        attn_output, attention_weights = self.mha.call(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        FROM TF
        :param d_model: dimension of model
        :param num_heads: number of heads fro multi-head attention
        :param dff: dimension of feed forward network
        :param rate: dropout rate
        """
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """
        :param x: inputs
        :param enc_output: output from encoder
        :param training: if in training
        :param look_ahead_mask: look ahead mask
        :param padding_mask: padding mask
        :return: output of decoder layer, attention weights from inputs,
            attention weights from enc_output TODO to check!
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1.call(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2.call(
          enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                   maximum_position_encoding, rate=0.1):
        """
        FROM TF
        :param num_layers: number of encoder layers
        :param d_model: dimension of model
        :param num_heads: number of attention heads
        :param dff: dimension of feed forward network
        :param input_vocab_size: vocab size of input
        :param maximum_position_encoding: maximal distance from positional encoding (Periode T)
        :param rate: dropout rate, defaults to 0.1
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        :param x: inputs
        :param training: if in training
        :param mask: padding mask
        :return: output of encoder final layer
        """

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.keras.backend.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        weights_all = []
        for i in range(self.num_layers):
          x, weights = self.enc_layers[i].call(x, training, mask)
          weights_all.append(weights)

        return x, weights_all  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                   maximum_position_encoding, rate=0.1):
        """
        FROM TF
        :param num_layers: number out decoder layers
        :param d_model: dimension of model
        :param num_heads: number of attention heads
        :param dff: dimension of feed forward network
        :param target_vocab_size: covab size of target space
        :param maximum_position_encoding: maximum distance (Periode T)
        :param rate: dropout rate, defaults to 0.1
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
               look_ahead_mask, padding_mask):
        """
        :param x: inputs
        :param enc_output: final output from encoder
        :param training: if in training
        :param look_ahead_mask: look ahead mask
        :param padding_mask: padding mask
        :return: final output of decoder, attention weights for decoder
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.keras.backend.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i].call(x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                   target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        FROM TF
        :param num_layers: number of encoder/decoder layers
        :param d_model: dimension of model
        :param num_heads: number of attention heads
        :param dff: dimension of feed forward network
        :param input_vocab_size: vocab size of input
        :param target_vocab_size: vocab size of output
        :param pe_input: maximal positional encoding value input
        :param pe_target: maximal positional encoding value target
        :param rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.dense_layer = tf.keras.layers.Dense(1)
        self.seq_class_layer = tf.keras.layers.Dense(1)
        #    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
        #                           target_vocab_size, pe_target, rate)

        #self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
               look_ahead_mask, dec_padding_mask):
        """
        TODO why isn't this outputting the weights?
        :param inp: input
        :param tar: input but shifted over (teacher forcing)
        :param training: if is training
        :param enc_padding_mask: encoding padding mask
        :param look_ahead_mask: look ahead mask
        :param dec_padding_mask: decoding padding mask

        NOTE this has been modified from the original to allow binary classification

        :return: logits representing the classification
        """

        enc_output, all_weights = self.encoder.call(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        mask = tf.math.logical_not(tf.math.equal(inp, 0))
        mask = tf.keras.backend.cast(mask, dtype=enc_output.dtype)
        length = tf.reduce_sum(mask, keepdims=True, axis=-1)
        #    y = tf.reduce_sum(mask[:, :, tf.newaxis] * enc_output, axis=1)
        #    y = tf.divide(y, length)
        y = mask[:, :, tf.newaxis] * enc_output

        y = self.dense_layer(y)
        y = tf.squeeze(y, [-1])
        logits = self.seq_class_layer(y)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        # dec_output, attention_weights = self.decoder(
        #    tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        #    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return tf.squeeze(logits), all_weights


# mostly copied from TF
num_layers = 1
d_model = 16
dff = 16 #512
num_heads = 1 #8

input_vocab_size = len(vocab) + 3   # 0, START, STOP
target_vocab_size = input_vocab_size
dropout_rate = 0.1

#
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         """
#         :param d_model: dimension of model
#         :param warmup_steps: how many steps over which not to change learning rate, defaults to 4000
#         """
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.keras.backend.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         """
#         :param step: current step number (in training
#         :return: the learning rate
#         """
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


"""
setting training parameters and variables
"""

# learning_rate = CustomSchedule(d_model)

# TODO this currently isn't using the learning rate above?
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# commented is from TF
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#    from_logits=True, reduction='none')
loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True)


def loss_function(real, pred):
    """
    FROM TF
    :param real: labels
    :param pred: outputs (predicted labels)
    :return: masked loss calculation
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


"""
initialize training variables
"""

train_loss = tf.keras.metrics.Mean(name='train_loss')
accuracy = tf.keras.metrics.Accuracy(
    name='accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=MAX_LENGTH,
                          pe_target=MAX_LENGTH,
                          rate=dropout_rate)


def create_masks(inp, tar):
    """
    FROM TF
    :param inp: input values
    :param tar: target values TODO is this outputs or labels? -- look @ line 512
    :return:
    """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


"""
saving checkpoints
"""
# TODO see if I can't get them to visualize? (use this as a way to get insight into process)

checkpoint_path = './checkpoints/76hhobbmw/'

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                           optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#    ckpt.restore(ckpt_manager.latest_checkpoint)
#    print('Latest checkpoint restored!!')

# TODO figure out where to use this
def plot_attention_weights(attention, sentence, result, layer):
    """
    :param attention: the attention weights outputted by the nn
    :param sentence: the input into the nn
    :param result: the output of the nn
    :param layer: layer to be displayed
    :return: a plot displaying ....
    """
    fig = plt.figure(figsize=(16, 8))

    print("vis")
    print(sentence[0])

#    sentence = encode_a_line(sentence)

    #attention = tf.squeeze(attention[layer], axis=0)
    attention = attention[0][0][0]
    # attention = tf.squeeze(attention, axis=0)
    print(attention)
    print(attention.shape)

    ax = fig.add_subplot(2, 4, 1)

    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 10}

    # ax.set_xticks(range(attention.shape[0] + 2))
    # ax.set_yticks(range(len(result)))
    #
    # ax.set_ylim(len(result) - 1.5, -0.5)
    #
    # ax.set_xticklabels(
    #     ['<start>'] + [decode_a_line([i]) for i in sentence] + ['<end>'],
    #     fontdict=fontdict, rotation=90)
    #
    # ax.set_yticklabels([decode_a_line().decode([i]) for i in result
    #                     if i < VOCAB_SIZE],
    #                    fontdict=fontdict)
    #
    # ax.set_xlabel('Head {}'.format(head + 1))

    # for head in range(attention.shape[0]):
    #     ax = fig.add_subplot(2, 4, head + 1)
    #
    #     # plot the attention weights
    #     ax.matshow(attention[head][:-1, :], cmap='viridis')
    #
    #     fontdict = {'fontsize': 10}
    #
    #     ax.set_xticks(range(len(sentence) + 2))
    #     ax.set_yticks(range(len(result)))
    #
    #     ax.set_ylim(len(result) - 1.5, -0.5)
    #
    #     ax.set_xticklabels(
    #         ['<start>'] + [decode_a_line([i]) for i in sentence] + ['<end>'],
    #         fontdict=fontdict, rotation=90)
    #
    #     ax.set_yticklabels([decode_a_line().decode([i]) for i in result
    #                         if i < VOCAB_SIZE],
    #                        fontdict=fontdict)
    #
    #     ax.set_xlabel('Head {}'.format(head + 1))
    #
    plt.tight_layout()
    plt.show()


def run_nn_step(inp, tar, label, is_training=True,vis=False):
    """
    this deviates from TF
    :param inp: bit-encoded input
    :param tar: TODO okay what is target - putting in same thing as for inputs (in 757) (@ 512)
    :param label: label
    :param is_training: whether is in training
    :return: predictions
    """

    if not tf.is_tensor(inp):
        inp = tf.convert_to_tensor(inp, dtype=tf.int32)
    if not tf.is_tensor(tar):
        tar = tf.convert_to_tensor(tar, dtype=tf.int32)
    if not tf.is_tensor(label):
        label = tf.convert_to_tensor(label, dtype=tf.float32)

    tar_inp = tar

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    if is_training:
        with tf.GradientTape() as tape:
            logits, all_weights = transformer.call(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = tf.keras.losses.binary_crossentropy(label, logits, from_logits=True)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_loss.update_state(loss)
    else:
        logits, all_weights = transformer.call(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        if vis:
            plot_attention_weights(all_weights, inp,None,None)

    # logits and sigmoid because classification
    predictions = tf.round(tf.nn.sigmoid(logits))

    accuracy.update_state(label, predictions)

    return predictions


def batch_data(dataset, batch_size):
    """
    own code
    TODO understand at what exactly batching does
    :param dataset: dataset
    :param batch_size: wanted batch size
    :return: the batched dataset
    """
    batched = []
    for i in range(0, len(dataset), batch_size):
        data_batch = []
        label_batch = []
        for j in range(batch_size):
            if i+j < len(dataset):
                data, label = dataset[i+j]
                data_batch.append(data)
                label_batch.append(label)
        batched.append((data_batch, label_batch))
    return batched




def testing_classification():
    """
    own code
    :return: prints accuracy of running testing
    """
    test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_label)).batch(BATCH_SIZE)
    accuracy.reset_states()
    for (batch, (inp, label)) in enumerate(test_dataset):
        #print(inp, label, run_nn_step(inp, inp, label, False))
        run_nn_step(inp, inp, label, False,vis=(batch % 20 == 0))
    print('Accuracy {:.4f}'.format(accuracy.result()))


"""
running training
"""

# 100
EPOCHS = 10

for epoch in range(EPOCHS):
    if not TRAIN:
        break

    start = time.time()
    train_loss.reset_states()
    accuracy.reset_states()
    random.shuffle(train_dataset)
    for (batch, (inp, label)) in enumerate(batch_data(train_dataset, BATCH_SIZE)):
        run_nn_step(inp, inp, label, True)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), accuracy.result()))

        #if batch % 100 == 0:
        #    ckpt_save_path = ckpt_manager.save()
        #    print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))




testing_classification()


"""__________________END OF ACTUAL CLASSIFICATION___________________"""


# # TODO figure out where to use this
# def plot_attention_weights(attention, sentence, result, layer):
#     """
#     :param attention: the attention weights outputted by the nn
#     :param sentence: the input into the nn
#     :param result: the output of the nn
#     :param layer: layer to be displayed
#     :return: a plot displaying ....
#     """
#     fig = plt.figure(figsize=(16, 8))
#
#     sentence = encode_a_line(sentence)
#
#     attention = tf.squeeze(attention[layer], axis=0)
#
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head + 1)
#
#         # plot the attention weights
#         ax.matshow(attention[head][:-1, :], cmap='viridis')
#
#         fontdict = {'fontsize': 10}
#
#         ax.set_xticks(range(len(sentence) + 2))
#         ax.set_yticks(range(len(result)))
#
#         ax.set_ylim(len(result) - 1.5, -0.5)
#
#         ax.set_xticklabels(
#             ['<start>'] + [decode_a_line([i]) for i in sentence] + ['<end>'],
#             fontdict=fontdict, rotation=90)
#
#         ax.set_yticklabels([decode_a_line().decode([i]) for i in result
#                             if i < VOCAB_SIZE],
#                            fontdict=fontdict)
#
#         ax.set_xlabel('Head {}'.format(head + 1))
#
#     plt.tight_layout()
#     plt.show()

