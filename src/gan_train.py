import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from gan_config import *

#---   データ読み込み
MD_DATA = np.loadtxt(data_name)

correct_disp = np.zeros(shape=(point_mol_num, data_step, dim))
correct_disp[0, :, 0] = np.array(MD_DATA[:data_step, 1])

data_of_MD = np.zeros(shape=(1, data_step, 1))
data_of_MD[0, :, 0] = np.array(MD_DATA[:data_step, 1])

TRAIN_DATA = np.loadtxt(data_name)
training_data = np.zeros(shape=(1, use_step, 1))
training_data[0, :, 0] = np.array(TRAIN_DATA[:use_step, 1])

del MD_DATA

AVERAGE_DATA = np.average(training_data)
STD_DATA = np.std(training_data)
training_data = (training_data - AVERAGE_DATA) / STD_DATA

info = pd.DataFrame(data=[["data_step", data_step], ["use_step", use_step], 
                          ["AVERAGE_OF_DATA", AVERAGE_DATA], ["STD_OF_DATA", STD_DATA]], 
                    columns=["parameter", "value"])

info = pd.concat([info, pd.DataFrame(data=[["sequence_length", sequence_length]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["iteration_all", iteration_all]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["batch_size", batch_size]], columns=["parameter", "value"])])

info = pd.concat([info, pd.DataFrame(data=[["random_uniform_inf", random_uniform_inf]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["means2", means2]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["random_uniform_sup", random_uniform_sup]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["std2", stds2]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["discriminator_extra_steps", discriminator_extra_steps]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["gen_lr", gen_lr]], columns=["parameter", "value"])])
info = pd.concat([info, pd.DataFrame(data=[["disc_lr", disc_lr]], columns=["parameter", "value"])])

def discriminator_loss(true_data, recon_data):
    return -(tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data))

def generator_loss(true_data, recon_data):
    return tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data)

generator_optimizer = keras.optimizers.RMSprop(learning_rate=gen_lr)
discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=disc_lr)

def build_generator():
    noise_inputs = tf.keras.Input(shape=(sequence_length, dim))
    c_noise = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal", padding='valid')(noise_inputs)
    flat_noise_g = tf.keras.layers.Flatten()(c_noise)
    innx = tf.keras.layers.Dense(units=sequence_length, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(flat_noise_g)
    nxout = tf.keras.layers.Dense(units=sequence_length, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(innx)
    nx = tf.keras.layers.Concatenate(axis=1)([innx, nxout])
    x1 = tf.keras.layers.Dense(hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(nx)
    x2 = tf.keras.layers.Dense(hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(x1)
    x3 = tf.keras.layers.Dense(hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(x2)
    decoded = tf.keras.layers.Dense(units=sequence_length * dim)(x3)
    return tf.keras.Model(noise_inputs, decoded, name="generator")

def build_discriminator():
    disc_inputs = keras.Input(shape=(sequence_length, dim))
    dc1 = layers.Conv1D(filters=hidden_node, kernel_size=sequence_length, strides=sequence_length, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal", padding='valid')(disc_inputs)
    flat = layers.Flatten()(dc1)
    dc2 = layers.Conv1D(filters=16, kernel_size=1, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal", padding='valid')(disc_inputs)
    flat2 = layers.Flatten()(dc2)
    concat_layer_disc = tf.keras.layers.Concatenate(axis=1)([flat, flat2])
    dd1 = layers.Dense(units=hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(concat_layer_disc)
    dd2 = layers.Dense(units=hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(dd1)
    dd3 = layers.Dense(units=hidden_node, activation=tf.keras.layers.LeakyReLU(alpha=0), kernel_initializer="he_normal")(dd2)
    disc_out = layers.Dense(1)(dd3)
    return keras.Model(disc_inputs, disc_out, name="discriminator")

generator = build_generator()
discriminator = build_discriminator()

generator.summary()
discriminator.summary()

def gradient_penalty(true_data, fake_data, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1], minval=0.0, maxval=1.0, dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data * alpha * diff
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.math.reduce_mean((norm) ** 2)
    return gp

train_d_loss = tf.keras.metrics.Mean()
train_g_loss = tf.keras.metrics.Mean()

@tf.function()
def train_step(train_sample):
    data = train_sample
    d_steps_add = discriminator_extra_steps
    gp_weight = 10
    for l in range(d_steps_add):
        with tf.GradientTape() as tape:
            noise_inputer = tf.random.normal([batch_size, sequence_length, dim], mean=means2, stddev=stds2)
            reconstruction = generator(noise_inputer, training=False)
            reconstruction = tf.reshape(reconstruction, [batch_size, sequence_length, dim])
            fake_logits_temp = discriminator(reconstruction, training=True)
            fake_logits = fake_logits_temp
            true_logits = discriminator(data, training=True)
            d_cost = discriminator_loss(true_logits, fake_logits)
            gp = gradient_penalty(data, reconstruction, batch_size)
            d_cost = tf.cast(d_cost, tf.float64)
            d_loss = d_cost + gp * gp_weight
        d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))
    with tf.GradientTape() as tape:
        noise_inputer = tf.random.normal([batch_size, sequence_length, dim], mean=means2, stddev=stds2)
        reconstruction = generator(noise_inputer, training=True)
        reconstruction = tf.reshape(reconstruction, [batch_size, sequence_length, dim])
        fake_logits_temp = discriminator(reconstruction, training=False)
        fake_logits = fake_logits_temp
        true_logits = discriminator(data, training=False)
        g_cost = generator_loss(true_logits, fake_logits)
        g_cost = tf.cast(g_cost, tf.float64)
        g_loss = g_cost
    gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    train_d_loss.update_state(d_loss)
    train_g_loss.update_state(g_loss)
    return

def train_model():
    gen_array = []
    iteration_list = []
    gL_list = []
    dL_list = []
    counter_for_iteration = 0
    count_data_set = 0
    save_count = 1
    for i in range(1, iteration_all + 1):
        train_data = []
        temp = []
        for j in range(data_length):
            temp.append(training_data[0, j * sequence_length:(j + 1) * sequence_length])
        train_data = np.array(temp)
        train_step(train_sample=train_data)
        count_data_set += 1
        average_d_loss = train_d_loss.result()
        average_g_loss = train_g_loss.result()
        print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}".format(i + 1, average_d_loss, average_g_loss))
        train_d_loss.reset_states()
        train_g_loss.reset_states()
        iteration_list.append(counter_for_iteration)
        gL_list.append(average_g_loss)
        dL_list.append(average_d_loss)
    generator.save(r"/home/kawaguchi/model/" + "test_" + str(save_count) + ".h5")
    gen_array.append(generator)
    save_count += 1

    # 学習曲線の描画
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.plot(iteration_list, dL_list, color="red", label="discriminator Loss")
    ax.plot(iteration_list, gL_list, color="green", label="generator Loss")
    ax.set_xlabel("iteration", fontsize=30)
    ax.set_ylabel("Loss", fontsize=30)
    ax.legend(fontsize=30)
    ax.minorticks_on()
    ax.tick_params(labelsize=30, which="both", direction="in")
    plt.tight_layout()
    plt.savefig(r"/home/kawaguchi/result/training_proceed.png")
    plt.show()

    return gen_array

# モデルの学習を実行
gen_array = train_model()