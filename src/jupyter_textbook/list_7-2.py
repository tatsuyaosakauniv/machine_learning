import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1) # (A)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# データのload
outfile = np.load('class_data.npz')
X_train = outfile['X_train']
T_train = outfile['T_train']
X_test = outfile['X_test']
T_test = outfile['T_test']
X_range0 = outfile['X_range0']
X_range1 = outfile['X_range1']

# データの図示
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], linestyle='none', marker='o', markeredgecolor='black', color=c[i], alpha=0.8)
    plt.grid(True)

# 乱数の初期化
np.random.seed(1)

# Sequential モデルの作成
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid', kernel_initializer='uniform'))    # (A)
model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))    # (B)
sgd = tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.0, decay=0.0, nesterov=False)    # (C)  !!! lr は昔のバージョンしか使えない
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])    # (D)

# 学習
startTime = time.time()
history = model.fit(X_train, T_train, epochs=1000, batch_size=100, verbose=0, validation_data=(X_test, T_test))    #  (E)

# モデル評価
score = model.evaluate(X_test, T_test, verbose=0)   # (F)
print('cross entropy {0:.2f}, accuracy {1:.2f}'.format(score[0], score[1]))
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))