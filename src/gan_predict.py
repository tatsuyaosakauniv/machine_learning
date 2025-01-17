import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from gan_config import *

######################################################################################
############################# 学習済みモデル呼び出し ##################################
######################################################################################
# モデルを既に学習済みならここを実行する．
gen_array = []
for i in range(1,2):

    #load generator
    generator = keras.models.load_model(r"/home/kawaguchi/model/"+"test_1205"+str(i)+".h5",compile = False)
    gen_array.append(generator)
    pass

print("gen_array:",len(gen_array))

#######################################################################################
############################    予測フェーズ開始  ####################################
#######################################################################################

############################################################
####################### サンプル生成 ########################
############################################################
prediction_times = 1    #サンプルの生成数（この数だけ一つのモデルがdata_stepのデータ長のトラジェクトリを生成する．）

data_num = int(data_step/(sequence_length)) #繰り返し予測回数

#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
info_ad = pd.DataFrame(data=[["data_num",data_num]],columns = columns2)
info = info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#-- モデルを一つ呼び出して，prediction_times個のトラジェクトリを生成する．--#-----------------------#-----------------------#-----------------------#-----------------------#
#モデル呼び出し
generator = gen_array[0]

#ガウス分布乱数呼び出し
noise_inputer = tf.random.normal([prediction_times, sequence_length, dim],mean = means2,stddev = stds2)

reconstruction_ini = generator.predict(noise_inputer)

reconstruction_ini = tf.reshape(reconstruction_ini,[prediction_times, sequence_length, dim])

orbit_per_onemol = reconstruction_ini

print("noise_inputer: ", np.shape(noise_inputer))
print("reconstruction_ini: ", np.shape(reconstruction_ini))

for j in range(data_num-1):     # ここのrangeを1減らして、予測を元データと同じ長さにした

    #サンプリング生成
    #ノイズ入力準備####################################
    noise_inputer = tf.random.normal([prediction_times, sequence_length, dim],mean = means2,stddev = stds2)
    ##################################################

    reconstruction = generator.predict(noise_inputer)

    reconstruction = tf.reshape(reconstruction,[prediction_times, sequence_length, dim])

    #データ追加#######################################
    orbit_per_onemol = np.concatenate([orbit_per_onemol,reconstruction],axis = 1)
    pass

#一つのモデルから得られるトラジェクトリ数をリストに収める．
orbits = orbit_per_onemol

#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#


orbits = np.array(orbits)

print("orbits: ", np.shape(orbits))
print("data_of_MD: ", np.shape(data_of_MD))

#domain knowledge による補正
orbits[:,:,0] = orbits[:,:,0]*STD_DATA+ AVERAGE_DATA

####################################