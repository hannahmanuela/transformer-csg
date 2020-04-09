import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import TransformerDiscriminator


START_TOKEN = "<START>"
STOP_TOKEN = '<STOP>'
BATCH_SIZE = 64
TRAIN = True
EPOCHS = 10

num_layers = 1
d_model = 10
dff = 60
num_heads = 1


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
VOCAB_SIZE = len(vocab) + 1
input_vocab_size = len(vocab) + 3    # 0, START, STOP
target_vocab_size = input_vocab_size
dropout_rate = 0.1


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


train_loss = tf.keras.metrics.Mean(name='train_loss')
accuracy = tf.keras.metrics.Accuracy(
    name='accuracy')

string_data = pad_data(string_data)

TRAIN_SIZE = len(string_data) * 1 // 15
print(TRAIN_SIZE)

training_data = string_data[:TRAIN_SIZE]
training_label = label[:TRAIN_SIZE]
testing_data = string_data[TRAIN_SIZE:]
testing_label = label[TRAIN_SIZE:]

MAX_LENGTH = max([len(line) for line in string_data])


train_dataset = list(zip(training_data, training_label))


transformer = TransformerDiscriminator.Transformer(num_layers, d_model, num_heads, dff,
                                                   input_vocab_size, target_vocab_size,
                                                   pe_input=MAX_LENGTH,
                                                   pe_target=MAX_LENGTH,
                                                   rate=dropout_rate)


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
        string_data.append([VOCAB_SIZE] + [vocab[w] for w in tmp[0]] + [VOCAB_SIZE + 1])
        # string_data.append([vocab[w] for w in tmp[0]])
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

TRAIN_SIZE = len(string_data)-5
print(TRAIN_SIZE)

training_data = string_data[:TRAIN_SIZE]
training_label = label[:TRAIN_SIZE]
testing_data = string_data[TRAIN_SIZE:]
testing_label = label[TRAIN_SIZE:]

MAX_LENGTH = max([len(line) for line in string_data])

train_dataset = list(zip(training_data, training_label))


def plot_attention_weights(attention, sentence, result, layer):
    """
    :param attention: the attention weights outputted by the nn
    :param sentence: the input into the nn
    :param result: the output of the nn
    :param layer: layer to be displayed
    :return: a plot displaying ....
    """

    print("vis")
    # print(np.array(attention).shape)

    # attention = tf.squeeze(attention[layer], axis=0)
    # final_atten = attention[1][0][0][-1]
    for i in range(sentence.shape[0]):

        fig = plt.figure(figsize=(16, 8))

        # attention: (num layers, batch size, num heads, line length, line length)
        plot_attention = attention[0][i][0]
        print(sentence[i])

        ax = fig.add_subplot(2, 4, 1)

        ax.matshow(plot_attention, cmap='viridis')

        fontdict = {'fontsize': 10}

        plt.tight_layout()
        plt.show()


def run_nn_step(inp, tar, label, is_training=True, vis=False):
    """
    this deviates from TF
    :param inp: bit-encoded input
    :param tar: input again - teacher forcing for training
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

    enc_padding_mask, combined_mask, dec_padding_mask = TransformerDiscriminator.create_masks(inp, tar_inp, 0)

    if is_training:
        with tf.GradientTape() as tape:
            logits, all_weights = transformer.call(inp, tar_inp, True, enc_padding_mask, combined_mask,
                                                   dec_padding_mask)
            loss = tf.keras.losses.binary_crossentropy(label, logits, from_logits=True)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            TransformerDiscriminator.optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_loss.update_state(loss)
    else:
        logits, all_weights = transformer.call(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        print(tf.round(tf.nn.sigmoid(logits)))
        if vis:
            plot_attention_weights(all_weights, inp, None, None)

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
            if i + j < len(dataset):
                data, label = dataset[i + j]
                data_batch.append(data)
                label_batch.append(label)
        batched.append((data_batch, label_batch))
    return batched


def testing_classification():
    """
    own code -- runs testing
    :return: nothing
    """
    test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_label)).batch(BATCH_SIZE)
    accuracy.reset_states()
    for (batch, (inp, label)) in enumerate(test_dataset):
        # print(inp, label, run_nn_step(inp, inp, label, False))
        run_nn_step(inp, inp, label, False, vis=True)
    print('Accuracy {:.4f}'.format(accuracy.result()))


"""
running training
"""


def run():

    for epoch in range(EPOCHS):
        if not TRAIN:
            break

        start = time.time()
        train_loss.reset_states()
        accuracy.reset_states()
        random.shuffle(train_dataset)
        for (batch, (inp, label)) in enumerate(batch_data(train_dataset, BATCH_SIZE)):
            if batch % 50 == 0:
                run_nn_step(inp, inp, label, True, vis=True)
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), accuracy.result()))
            else:
                run_nn_step(inp, inp, label, True)

            # if batch % 100 == 0:
            #    ckpt_save_path = ckpt_manager.save()
            #    print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    testing_classification()

run()

