import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(1, figsize=(12, 3.2))
plt.subplots_adjust(wspace=0.5)
plt.gray()

for id in range(3):
    plt.subplot(1, 3, id+1)
    img = x_train[id, :, :]
    plt.pcolor(255 - img)
    plt.text(24.5, 26, "%d" % y_train[id], color='cornflowerblue', fontsize=18)
    plt.xlim(0, 27)
    plt.ylim(27, 0)

plt.show()

x_train = x_train.reshape(60000, 784)   # (A)
x_train = x_train.astype('float32')     # (B)
x_train /= 255                          # (C)
num_classes = 10
y_train = to_categorical(y_train, num_classes)    # (D)

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test, num_classes)

np.random.seed(1)                                                                         # (A)

model = Sequential()                                                                      # (B)
model.add(Dense(16, input_dim=784, activation='sigmoid'))                                 # (C)
model.add(Dense(10, activation='softmax'))                                                # (D)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])    # (E)

startTime = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=1000, verbose=1, validation_data=(x_test, y_test))  # (A)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:3f} sec".format(time.time() - startTime))