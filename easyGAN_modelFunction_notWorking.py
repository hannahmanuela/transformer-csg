import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


BATCH_SIZE = 250
NUM_EXS = 200
train_steps = 10000
SEED_SIZE = 100


def sample_data(num_exs, myRange=100):
    data = []
    x = myRange * (np.random.random_sample((num_exs,)) - 0.5)

    for i in range(num_exs):
        val = int(x[i])
        data.append([val, val*val])

    return data


def rand_sample(m):
    return np.random.randint(low=-100, high=100, size=m)


LAYER_DIMS = 16

discriminator = Sequential()
discriminator.add(tf.keras.layers.Dense(LAYER_DIMS, activation=tf.nn.leaky_relu, input_shape=SEED_SIZE))
discriminator.add(Dense(LAYER_DIMS, activation=tf.nn.leaky_relu))
discriminator.add(Dense(1))

generator = Sequential()
generator.add(Dense(LAYER_DIMS, activation=tf.nn.leaky_relu))
generator.add(Dense(LAYER_DIMS, activation=tf.nn.leaky_relu))
generator.add(Dense(2))


def discriminator_loss(real_output, fake_output):
    real_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
    fake_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)


@tf.function
def train_step(dataSet):
    vis = True
    seed = tf.convert_to_tensor(rand_sample(SEED_SIZE))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_dataPoints = generator.predict(seed, steps=1)

        real_output = discriminator.predict(dataSet, steps=1)
        fake_output = discriminator.predict(generated_dataPoints, steps=1)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        if vis:
            plt.scatter(*zip(*dataSet))
            plt.scatter(*zip(*generated_dataPoints), c='red')
            plt.plot()
            plt.show()

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(epochs):

    for epoch in range(epochs):

        real_exs = tf.convert_to_tensor(sample_data(BATCH_SIZE))

        if epoch % 1000 == 0:
            g_loss, d_loss = train_step(real_exs)
        else:
            g_loss, d_loss = train_step(real_exs)

        log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
        log_mesg = "%s  [G loss: %f, acc: %f]" % (log_mesg, g_loss[0], g_loss[1])
        print(log_mesg)


train(train_steps)



# adversarial = Sequential()
# adversarial.add(generator)
# adversarial.add(discriminator)


# generator.compile(loss="binary_crossentropy", metrics=['accuracy'])
# discriminator.compile(loss="binary_crossentropy", metrics=['accuracy'])
# adversarial.compile(loss="binary_crossentropy", metrics=['accuracy'])

# for i in range(train_steps):
#
#     real_exs = sample_data(BATCH_SIZE)
#     rand_seeds = rand_sample(BATCH_SIZE, 2)
#
#     gen_exs = generator.predict(rand_seeds)
#
#     if i % 1000 == 0:
#         plt.scatter(*zip(*real_exs))
#         plt.scatter(*zip(*gen_exs), c='red')
#         plt.plot()
#         plt.show()
#
#     x = np.concatenate((real_exs, gen_exs))
#     y = np.ones([2 * BATCH_SIZE, 1])
#     y[BATCH_SIZE:, :] = 0
#     d_loss = discriminator.train_on_batch(x, y)
#
#     y = np.ones([BATCH_SIZE, 1])
#     rand_seeds = rand_sample(BATCH_SIZE, 2)
#     a_loss = adversarial.train_on_batch(rand_seeds, y)
#     log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
#     log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
#     print(log_mesg)
