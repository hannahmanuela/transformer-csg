import tensorflow as tf
import numpy as np


"""_______ACTUAL TRANSFORMER - PROCESSING DATA________"""


def get_angles(pos, i, d_model):
    """
    FROM TF
    :param pos: position values for each data point
    :param i: dimension of model currently in???
    :param d_model: dimension of model - size of layers within the model
    :return: angle for positional encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def create_padding_mask(seq, eq_to):
    """
    FROM TF
    :param seq: input values
    :return: indicates where pad value 0 is present:
        it outputs a 1 at those locations, and a 0 otherwise
    """

    seq = tf.cast(tf.math.equal(seq, eq_to), tf.float32)

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
# pos_encoding = positional_encoding(50, 512)


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
        self.d_model = d_model

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

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

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
            attention weights from enc_output
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

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape = (batch_size, target_seq_len, d_model)
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

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        """
        :param inp: input
        :param tar: input but shifted over (teacher forcing)
        :param training: if is training
        :param enc_padding_mask: encoding padding mask
        :param look_ahead_mask: look ahead mask
        :param dec_padding_mask: decoding padding mask

        NOTE this has been modified from the original to allow binary classification

        :return: logits representing the classification
        """

        enc_output, all_weights = self.encoder.call(inp, training,
                                                    enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        mask = tf.math.logical_not(tf.math.equal(inp, 0))
        mask = tf.keras.backend.cast(mask, dtype=enc_output.dtype)
        y = mask[:, :, tf.newaxis] * enc_output  # (batch_size, input_seq_len, d_model)
        y = self.dense_layer(y)

        y = tf.squeeze(y, [-1])

        logits = self.seq_class_layer(y)

        return tf.squeeze(logits), all_weights


"""
setting training parameters and variables
"""

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

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


def create_masks(inp, tar, padding_char):
    """
    FROM TF
    :param inp: input values
    :param tar: input values - teacher forcing
    :param padding_char: the char the padding consists of
    :return: all the masks (encoding, combined, decoding)
    """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, padding_char)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp, padding_char)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, padding_char)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
