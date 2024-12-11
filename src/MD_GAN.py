######################################################################################
#################################### 初期設定 ########################################
######################################################################################

#---   ライブラリの設定（色々弄っていく中で要らなくなったものも混じってる．）
import numpy as np
import scipy as sp
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import os
import time
import pandas as pd
#------------------------------------------------------------------------------------

#---   計算前のワークステーション設定

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #GPUが使うメモリの指定．ない場合，一回のプログラム実行でGPUのメモリをすべて使う．

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #g01のワークステーションは明示的にどのGPUを使うかを指定しないといけない．
                                                           #環境変数で設定する．

#------------------------------------------------------------------------------------

#---   GPU or CPU の指定
def use_cpu():
    #CPU計算をしたい時
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #環境変数を指定するだけ． 例えば，GPUマシンが埋まっているとか，学習済みモデルを使うだけの時なんかはCPUだけ使う方が早い事もある．
    return
def use_gpu():                                   #デフォルトではGPUを使う事になっている．
    #GPU計算をしたい時
    return

use_gpu()

#------------------------------------------------------------------------------------

#---   上の指定を経たうえでインポート出来るライブラリ群 #注意点として，import keras とimport tensorflow.kerasでは枠組みが異なるので，互換が効かない場合がある．
                                                     #どちらかに統一する事を推奨するが，個人的にはtensorflowが提供するkeras以外の関数なども使うのでimport tensorflow.kerasを使う方がオススメ．
                                                     #もし混ぜこぜを直すのが面倒なら，下記のようにfrom tensorflow import kerasにするとkeras.○○とかもtensorflow.keras.○○として扱ってくれるので楽．

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

#------------------------------------------------------------------------------------

#---   図のデフォルト設定の変更
from matplotlib.ticker import ScalarFormatter

class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                useMathText=useMathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._order_of_mag

# いつかは出来るようにしたいけど，linuxでのTimes New Romanでの描画はFontがないですって言われる．一応エラー文みたいなのが出るけど問題なく回る．
# TImes New Romanを使いたくて色々やってたら仮想環境が全部吹き飛んだので諦める．キレそう．
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"]="stix"

#------------------------------------------------------------------------------------

#---   seed固定 (多分完全にシード値の固定をして再現性を保証する事は出来ないと思われるが，気休め程度に固定する．)

def set_seed(seed):

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    tf.random.set_seed(seed)
    return

set_seed(1)

#------------------------------------------------------------------------------------

#---   データ読み込み及び必要なパラメータの指定

#フォルダとファイル名指定及びその読み込み
address = r"/home/s_tanaka/result/pat8" + "/"               #r"[ファイルが入ってるフォルダー名]"+"/"

DATA_filename = "temp_ar1.dat" 
data_name = address + DATA_filename

MD_DATA = np.loadtxt(data_name)


#!!!parameters
dt = 2.0 #時間刻み [fs]     #後々pythonで物理量の評価もしたいなら必要
fs = 1.0E-15

#------------------------------------------------------------------------------------


#---   データ読み込み及び必要なパラメ―タ処理2 (主に機械学習でどれだけデータを使うかなどを指定する．)
#データ前処理用の色々
#!!!parameters
data_step = 60000 #MDのサンプルから取り出してくるデータ長
use_step = 20000   #学習に使うデータ長

#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
columns2 = ["parameter","value"]
info = pd.DataFrame(data = [["data_step",data_step]],columns= columns2)
info_ad = pd.DataFrame(data=[["use_step",use_step]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#予測データとの比較用のMDデータ成形（つまり，test data）-----------
print("MD_DATA shape: ",np.shape(MD_DATA))

data_of_MD = np.zeros(shape=(1,data_step,1))
data_of_MD[0,:,0] = np.array(MD_DATA[:data_step,2]) #基本的に，配列はndarray型にする方が楽であるが，list型の方がメモリ量は少なく済むので，大きいデータを扱う時は要注意．


#学習データの成形（つまり，train data）-----------
#学習用トラジェクトリデータ
DATA_filename = "temp_ar1.dat"                   #学習用ファイル
data_name = address + DATA_filename

TRAIN_DATA = np.loadtxt(data_name)

#学習データ成形-----------
training_data = np.zeros(shape=(1,use_step,1))

training_data[0,:,0] = np.array(TRAIN_DATA[:use_step,2])

del MD_DATA     #メモリ不足が心配なので，デカいメモリ持ってそうな変数は動的にメモリ開放．なお，60000くらいのサイズなら全然大きくない．

#------------------------------------------------------------------------------------


#---   データ前処理 （標準化処理）
AVERAGE_DATA = np.average(training_data)
STD_DATA = np.std(training_data)
training_data = (training_data-AVERAGE_DATA)/STD_DATA
# ↑ ------------------------------------------------------ まとめて標準化しちゃだめな気がする


#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
info_ad = pd.DataFrame(data=[["AVERAGE_OF_DATA",AVERAGE_DATA]],columns = columns2)
info = pd.concat([info,info_ad])
info_ad = pd.DataFrame(data=[["STD_OF_DATA",STD_DATA]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#------------------------------------------------------------------------------------


#---   データ前処理2(学習データ周りのパラメータなどを設定．)
""" 変数の説明
sequence_length      : 一回の予測で生成されるデータ長
data_length : 一つの分子からとれるデータサンプル長 （学習用のデータ長/一回の予測のデータ長）
iteration_all        : 勾配を更新する回数の総合
batch_size           : バッチ数
"""

#!!!parameters
sequence_length =500
batch_size = int(use_step/sequence_length)


data_length = int(use_step/sequence_length)
iteration_all = 20000
#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
info_ad = pd.DataFrame(data=[["sequence_length",sequence_length]],columns = columns2)
info = pd.concat([info,info_ad])
info_ad = pd.DataFrame(data=[["iteration_all",iteration_all]],columns = columns2)
info = pd.concat([info,info_ad])
info_ad = pd.DataFrame(data=[["batch_size",batch_size]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#------------------------------------------------------------------------------------

######################################################################################
################################## 初期設定終了 #######################################
######################################################################################


#######################################################################################
################################# モデル訓練パート #####################################
#######################################################################################
#--- パラメータ指定 1段階目
""" 変数の説明
menas                    :潜在変数生成機の入力乱数のガウス分布における平均値指定．  訳あって使わなくなった．
stds                     :潜在変数生成機の入力乱数のガウス分布における標準偏差指定．訳あって使わなくなった．
random_uniform_inf       :潜在変数生成機の入力乱数の一様分布における下限． ↑のガウス分布の代わりに使うようになった
random_uniform_sup       :潜在変数生成機の入力乱数の一様分布における上限． ↑のガウス分布の代わりに使うようになった
menas2                   :生成機の入力乱数のガウス分布における平均値指定
stds2                    :生成機の入力乱数のガウス分布における標準偏差指定．

dim                      :MDデータの次元xyz

discriminator_extra_steps:判別機の学習回数/生成機の学習回数．   は生成機より余分に学習を行う．（その代わり学習率を下げる．）それがWGANでは一般的．

gen_lr                   :生成機の学習率
disc_lr                  :判別機の学習率

Normalize_axis           :潜在変数のにかける制約の方向（0でBatch_normalize, 1でLayer_normalize）
"""
#!!!parameters

# menas = 0
# stds = 0.2
random_uniform_inf = 0
random_uniform_sup = 1.0
menas2 = 0              # meansとしたつもりが menasになってました．リトアニア語で"美術"だそうです
stds2 = 1.0

dim = 1


discriminator_extra_steps = 5
gen_lr = 0.000002
disc_lr = 0.0000004

#------------------------------------------------------------------------------------

#--- 損失関数，最適化手法決定
# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to discriminator loss.
def discriminator_loss(true_data, recon_data):
    return -(tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data))

# Define the loss functions for the generator. 誤差の最大化をする処理はtensorflowにはない．https://www.brainpad.co.jp/doors/contents/01_tech_2017-09-08-140000/
#                                              そのため，正負の反転で対応することになる．
def generator_loss(true_data,recon_data): 
    return tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data)


#WGAN-GPではAdamを，WGANではRMSpropを使うらしいが，自分の場合はRMSpropの方が上手くいっているのでこちらを使う．正直，RMSpropとAdamの優劣はつかない気がする．

generator_optimizer = keras.optimizers.RMSprop(
    learning_rate = gen_lr
)

discriminator_optimizer = keras.optimizers.RMSprop(
    learning_rate = disc_lr
)

#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
info_ad = pd.DataFrame(data=[["random_uniform_inf",random_uniform_inf]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["means2",menas2]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["random_uniform_sup",random_uniform_sup]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["std2",stds2]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["discriminator_extra_steps",discriminator_extra_steps]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["gen_lr",gen_lr]],columns = columns2)
info = pd.concat([info,info_ad])

info_ad = pd.DataFrame(data=[["disc_lr",disc_lr]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#------------------------------------------------------------------------------------


#--- モデル構築

#自作の色々欄
def Linear_tanh(z,alpha = 0.16):            #MLpotentialなんかではこういう活性化関数が使われる．試しに使ってみたけど中々よさそう．
    return alpha*z + tf.math.tanh(z)

 # !!!!!!!!!!!!--------------------こっちのコードでは潜在変数を使ってないので注意
 
#潜在変数にかける補正．要らない時はreturnとdef部以外をコメントアウト． ちなみにbatch数1とかではNormalize_axis=0は使うべきではない．
def normalize_latent(latent):
    # latent_mean = tf.math.reduce_mean(latent,axis=Normalize_axis,keepdims=True)
    # std_mean = tf.math.reduce_std(latent,axis=Normalize_axis,keepdims=True)
    
    # latent = (latent-latent_mean)/std_mean
    return latent

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#学習済みモデル呼び出し時はここから「学習モデル呼び出し」までをコメントアウト．
#########################
#### build generator ####
#########################

#-- input
noise_inputs = tf.keras.Input(shape = (sequence_length,dim))#ガウスノイズ
c_noise = tf.keras.layers.Conv1D(filters = 32, kernel_size = 1,strides = 1,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(noise_inputs)
flat_noise_g = tf.keras.layers.Flatten()(c_noise)
innx = tf.keras.layers.Dense(units = sequence_length,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(flat_noise_g)
# mapping
nxout = tf.keras.layers.Dense(units = sequence_length,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(innx)
nx = tf.keras.layers.Concatenate(axis=1)([innx,nxout])

#-- hidden layers
x1 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(nx)
x2 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x1)
x3 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x2)

#-- output
decoded = tf.keras.layers.Dense(units = sequence_length*dim)(x3)

generator = tf.keras.Model(noise_inputs,decoded, name = "generator")

generator.summary()

#############################
#### build discriminator ####
#############################

#-- input
disc_inputs = keras.Input(shape = (sequence_length,dim))

#-- hidden layers
dc1 = layers.Conv1D(filters = 2048, kernel_size = sequence_length,strides = sequence_length,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
flat = layers.Flatten()(dc1)    # 多次元を1次元にする関数


#
dc2 = layers.Conv1D(filters = 16, kernel_size = 1,strides = 1,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
flat2 = layers.Flatten()(dc2)

#
# dc3 = layers.Conv1D(filters = 1024, kernel_size = 96,strides = 96,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
# flat3 = layers.Flatten()(dc3)

# #
# dc4 = layers.Conv1D(filters = 512, kernel_size = 48,strides = 48,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
# flat4 = layers.Flatten()(dc4)

concat_layer_disc = tf.keras.layers.Concatenate(axis=1)([flat,flat2])
#
dd1 = layers.Dense(units = 2048,activation = layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(concat_layer_disc)
dd2 = layers.Dense(units = 2048,activation = layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(dd1)
dd3 = layers.Dense(units = 2048,activation = layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(dd2)

#-- output
disc_out = layers.Dense(1)(dd3)

discriminator = keras.Model(disc_inputs,disc_out, name = "discriminator")

discriminator.summary()
##########################################################

"""
GAN定義開始 Reference : https://www.tensorflow.org/guide/keras/custom_layers_and_models?hl=ja#putting_it_all_together_an_end-to-end_example

一口メモ

書き方をpytorchに寄せる．要するにdefine by Runで学習を行うという事．fitにデータを入れるやつはdefine and run．
違いとしては
予め計算のシステムを作り上げて，そこにデータを流し込むやり方→define and run
データを生み出しながら正解と比較していくやり方→define by run

今はその間位の使い方をしてる．

MD-GANは恐らく，後者に特化した作り方．論文の書き方を見るに，開発者はpyTorchで書いてるはず．
Pytorchはデフォルトでdefine by runだが，Tensorflowはどちらもできる．（但し，広く知れ渡ってるのはdefine and run）

TensorflowにあるEager Executionというもの．これがdefine by runをtensorflowで可能にするインターフェース．
ver1.5まではsess.run()とかの宣言が必要だったとか．その後ver2.0を出してシンプルになったとか．自分はver2.0で書いている．

書き方はここと公式のリファレンスを見ながら
https://wshinya.hatenablog.com/entry/2019/10/18/144025
https://www.tensorflow.org/guide/function?hl=ja

@tf.functionをつけるとgraphモード               (define and run)
@tf.fuctionなし(コメントアウト)だとEagerモード   (define by run) 

要するに，@tf.functionとすると，学習を効率的に行うためのデータの通り道（グラフ）が作られ，その通り道を使いまわす．なしだと，毎回道を作るイメージ．
前者はメモリを多く使うが，早い．後者はメモリを圧迫しないが，遅い．

1. 初期値生成（論文に準拠）．Snz(一様乱数 [0,1])を初期値として，分布緩和を導入．7~10回の繰り返し生成で分布がおよそ一定になるらしい．

2. iteration回の勾配更新実行．(train_step関数呼び出し)
    2-a. 前回の学習もしくは1.で生成した潜在変数を受け取る．
    2-b. discriminatorの学習(WGANのリファレンスコードではgeneratorの5倍ほど学習回数を増やし，その代わり学習率を1/5にしており，これが一般的っぽい)
    2-d. generatorの学習

3. d_lossとかの表示の奴をリセットする．

"""

#--- 学習周りの定義
train_d_loss = tf.keras.metrics.Mean()
train_g_loss = tf.keras.metrics.Mean()


#  訓練過程の定義 （WGANのリファレンスコードでは，生成機用のGPはない．自分で付け足した感じ．）

#GP_penaltyの実装
def gradient_penalty(true_data,fake_data,batch_size):
    alpha = tf.random.uniform([batch_size,1,1],minval = 0.0,maxval = 1.0,dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data*alpha*diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated,training = True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp

#GP_penaltyの実装(生成機用)
def gradient_penaltyG(true_data,fake_data,batch_size):
    alpha = tf.random.uniform([batch_size,1,1],minval = 0.0,maxval = 1.0,dtype=tf.float64)
    true_data = tf.cast(true_data, tf.float64)
    fake_data = tf.cast(fake_data, tf.float64)
    diff = fake_data - true_data
    interpolated = true_data*alpha*diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated,training = True)
    
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis = [1,2]))
    gp = tf.math.reduce_mean((norm)**2)
    return gp




#学習過程の記述
"""
一口解説
データを３回生成し，その３回分一纏めのデータ長で相似具合を比較する．

コーディングの際には，次のように行う．↓

with tf.GradientTape() as tape:
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    データを生成する及び，誤差の計算
    実際の使い方ははtrain_stepを見る事を推奨
    ~~~~~~~~~~~~~~~~~~~~~~~~~

tape.Gradientで↑のtapeの勾配を作る．
apply_gradientで勾配の更新を実行．

with文はpythonを使っててもあまり見る事はないし，自分もよく分かってはいないけれど，
ネット上で軽く調べた感じログをファイルとして保存し，apply_gradientを実行した時に消去してる
とかそんな感じな気がする．


自分のtrain_step内では次の順序でモデルの更新を行う．
１．discriminatorを訓練．
３．生成機を訓練

同時に更新する事も出来る．
"""

@tf.function()
def train_step(train_sample):
    data = train_sample
    d_steps_add = discriminator_extra_steps
    gp_weight = 10  #GPの係数．式中のλ.

    ########################
    # discriminator学習開始 #
    ########################
    for l in range(d_steps_add):
        with tf.GradientTape() as tape :
            #データ生成
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction = generator(noise_inputer,training = False)
            
            reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])

            #偽データの評価値生成
            fake_logits_temp = discriminator(reconstruction,training = True)
            fake_logits = fake_logits_temp
            
            true_logits = discriminator(data,training = True)
            d_cost = discriminator_loss(true_logits,fake_logits)
            gp =  gradient_penalty(data,reconstruction,batch_size)
            d_cost = tf.cast(d_cost, tf.float64)
            d_loss = d_cost + gp*gp_weight
            
        d_gradient = tape.gradient(d_loss,discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(
            zip(d_gradient,discriminator.trainable_variables)
        )
    
    ########################################
    # generator学習開始 #
    ########################################
    with tf.GradientTape() as tape :

        #データ生成
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################

        reconstruction = generator(noise_inputer,training = True)
        
        reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])

        #偽データの評価値生成
        fake_logits_temp = discriminator(reconstruction,training = False)
        fake_logits = fake_logits_temp
        true_logits = discriminator(data,training = False)

        g_cost = generator_loss(true_logits,fake_logits)

        gp =  gradient_penaltyG(data,reconstruction,batch_size)
        g_cost = tf.cast(g_cost, tf.float64)
        g_loss = g_cost - gp*gp_weight
        
    gen_gradient = tape.gradient(g_loss,generator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gen_gradient,generator.trainable_variables)
    )

    train_d_loss.update_state(d_loss)
    train_g_loss.update_state(g_loss)

    return
##############################################################################################
##############################################################################################
##############################################################################################

gen_array = []                          #モデル保存用の箱

iteration_list = []
gL_list = []
latent_gL_list = []
dL_list = []


########################
#####   訓練実行   #####
########################

counter_for_iteration = 0
count_data_set = 0  #学習データは30000step(use_step)．学習を時系列的に繋げて行うので，use_step/(sequence_length*3)-2個のデータが出来上がる．
                    #時系列的なデータを使い切ったら，分子番号をランダムに指定してデータをピックアップし直す．そのフラグ管理用の変数．

#-- training
save_count = 1
counter = 0
time_start = time.time()

if(count_data_set == 0):
        train_data = []

        temp = []
        for j in range(data_length):
            temp.append(training_data[0,j*sequence_length:(j+1)*sequence_length,:])
        train_data = np.array(temp)
        pass
print(np.shape(training_data))
print(np.shape(train_data))

for i in range(1,iteration_all+1):
    train_data = []

    temp = []
    for j in range(data_length):
        temp.append(training_data[0,j*sequence_length:(j+1)*sequence_length,:])
    train_data = np.array(temp)
    pass
    
    train_step(train_sample = train_data)
    count_data_set += 1

    #loss 出力
    average_d_loss = train_d_loss.result()
    average_g_loss = train_g_loss.result()

    #loss画面表示
    print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}".format(i+1,average_d_loss,average_g_loss))

    #loss値のリセット
    train_d_loss.reset_states()
    train_g_loss.reset_states()

    #学習曲線のリストメイク
    iteration_list.append(counter_for_iteration)
    gL_list.append(average_g_loss)
    dL_list .append(average_d_loss)

    pass

generator.save(r"/home/s_tanaka/MD-GAN/test/models/"+"MD_GAN_gen_"+str(save_count)+".h5")
gen_array.append(generator)
save_count +=1

#-- 学習終了

#学習曲線 描画

#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

#plot
ax.plot(iteration_list,dL_list,color = "red",label  = "discriminator Loss")
ax.plot(iteration_list,gL_list,color = "green",label  = "latent generator Loss")

ax.set_xlabel("iteration",fontsize = 30)
ax.set_ylabel("Loss",fontsize = 30)

ax.legend(fontsize = 30)

ax.minorticks_on()

ax.tick_params(labelsize = 30, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/s_tanaka/MD-GAN/test/trainin_proceed.png")
plt.close()

#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
info_ad = pd.DataFrame(data=[["passed model",len(gen_array)]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

######################################################################################
############################# 学習済みモデル呼び出し ##################################
######################################################################################
# # モデルを既に学習済みならここを実行する．
# gen_array = []
# for i in range(1,2):

#     #load generator
#     generator = keras.models.load_model(r"/home/s_tanaka/MD-GAN/test/models/"+"MD_GAN_gen_"+str(i)+".h5",compile = False)
#     gen_array.append(generator)
#     pass

# print("gen_array :",len(gen_array))

#######################################################################################
############################    予測フェーズ開始  ####################################
#######################################################################################

t = 2.0 #時間刻み [fs]

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
noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)

reconstruction_ini = generator.predict(noise_inputer)

reconstruction_ini = tf.reshape(reconstruction_ini,[prediction_times,sequence_length,dim])

orbit_per_onemol = reconstruction_ini


for j in range(data_num):

    #サンプリング生成
    #ノイズ入力準備####################################
    noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)
    ##################################################

    reconstruction = generator.predict(noise_inputer)

    reconstruction = tf.reshape(reconstruction,[prediction_times,sequence_length,dim])

    #データ追加#######################################
    orbit_per_onemol = np.concatenate([orbit_per_onemol,reconstruction],axis = 1)
    pass

#一つのモデルから得られるトラジェクトリ数をリストに収める．
orbits = orbit_per_onemol
#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------#


orbits = np.array(orbits)

print(np.shape(orbits))
print(np.shape(data_of_MD))

#domain knowledge による補正
orbits[:,:,0] = orbits[:,:,0]*STD_DATA+ AVERAGE_DATA

####################################
"""
この行時点でのトラジェクトリの形状を開設する．

orbits[分子番号,余計な項,分子の軌跡,xyz]

・分子番号 ; 分子の番号．
・余計な項 ; 生成モデルに入力する際にCNNを使うとき，(200,)は受け取れない．CNN入力層の入力サイズは(200,1)の形状を取っている必要がある．その名残
・分子の軌跡 ; 予測したトラジェクトリ
・xyz ; 訓練集合と同次元データの，xyz次元

ここからまず一旦余計な項を除き，x,y,zに振り分ける．
arrayを仮のorbits保管庫として扱っている．後々にorbitsを使いたいときはこれを使う．


correct_disp[分子番号,xyz,分子の軌跡]

なお，温度を計算するタイミングで，correct_disp[分子番号，分子の軌跡，xyz]になる．

"""


# #学習データに使ったデータのステップ数
# boundary_for_graph = use_step

# #図示の際に表示する区間
# show_step = 3000

# ########以下，図示用の処理

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #prediction displacement------------------------
# time_step = np.arange(1,np.shape(orbits[i])[0]+1)
# ax.plot(time_step,orbits[i])
# ax.plot(time_step,correct_disp[0,0,:],color = "red")


# ax.set_xlabel("step",fontsize = 30)
# ax.set_ylabel("velocity",fontsize = 30)

# # ax.legend(fontsize = 30)

# ax.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/"+Ar_displacement_filename[:-4]+".png")
# plt.close()

# ########################
# #####  Green-Kubo  #####
# ########################

# print(np.shape(correct_disp))
# print(np.shape(orbits))


# correct_GK_x = np.zeros((nmsdtime))
# correct_GK_y = np.zeros((nmsdtime))
# correct_GK_z = np.zeros((nmsdtime))

# orbits_GK_x = np.zeros((nmsdtime))
# orbits_GK_y = np.zeros((nmsdtime))
# orbits_GK_z = np.zeros((nmsdtime))



# for j in range(n_picking):
#     correct_GK_x = correct_GK_x + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,0]*np.broadcast_to(correct_disp[:,j*shift_msd,0][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,0]))),axis = 0)/n_picking
#     correct_GK_y = correct_GK_y + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,1]*np.broadcast_to(correct_disp[:,j*shift_msd,1][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,1]))),axis = 0)/n_picking
#     correct_GK_z = correct_GK_z + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,2]*np.broadcast_to(correct_disp[:,j*shift_msd,2][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,2]))),axis = 0)/n_picking

#     orbits_GK_x = orbits_GK_x + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,0]*np.broadcast_to(orbits[:,j*shift_msd,0][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,0]))),axis = 0)/n_picking
#     orbits_GK_y = orbits_GK_y + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,1]*np.broadcast_to(orbits[:,j*shift_msd,1][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,1]))),axis = 0)/n_picking
#     orbits_GK_z = orbits_GK_z + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,2]*np.broadcast_to(orbits[:,j*shift_msd,2][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,2]))),axis = 0)/n_picking
#     pass

# ####

# time = np.arange(1,nmsdtime+1)*stepskip*dt*10**(-3)

# correct_GK_x = correct_GK_x*10**10
# correct_GK_y = correct_GK_y*10**10
# correct_GK_z = correct_GK_z*10**10

# correct_GK = (correct_GK_x+correct_GK_y+correct_GK_z)/3

# orbits_GK_x = orbits_GK_x*10**10
# orbits_GK_y = orbits_GK_y*10**10
# orbits_GK_z = orbits_GK_z*10**10

# orbits_GK = (orbits_GK_x+orbits_GK_y+orbits_GK_z)/3

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,correct_GK_x)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_x.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,orbits_GK_x)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_x.png")
# plt.close()



# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,correct_GK_y)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK y [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_y.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,orbits_GK_y)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_y.png")
# plt.close()


# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,correct_GK_z)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK z [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_z.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,orbits_GK_z)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_z.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,correct_GK,color="red")


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"correct_GK.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,orbits_GK)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"pred_GK.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,orbits_GK,color="blue")
# plt.plot(time,correct_GK,color="red")

# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"pred_and_correct_GK.png")
# plt.close()



# GK_int_correct_x = np.zeros((nmsdtime-1))
# GK_int_correct_y = np.zeros((nmsdtime-1))
# GK_int_correct_z = np.zeros((nmsdtime-1))

# GK_int_orbits_x = np.zeros((nmsdtime-1))
# GK_int_orbits_y = np.zeros((nmsdtime-1))
# GK_int_orbits_z = np.zeros((nmsdtime-1))

# for i in range(0,nmsdtime-1-1):
#     GK_int_correct_x[i+1] = GK_int_correct_x[i] + ((correct_GK_x[i]+correct_GK_x[i+1])/2.0)*dt*stepskip*10**(-15)
#     GK_int_correct_y[i+1] = GK_int_correct_y[i] + ((correct_GK_y[i]+correct_GK_y[i+1])/2.0)*dt*stepskip*10**(-15)
#     GK_int_correct_z[i+1] = GK_int_correct_z[i] + ((correct_GK_z[i]+correct_GK_z[i+1])/2.0)*dt*stepskip*10**(-15)

#     GK_int_orbits_x[i+1]  = GK_int_orbits_x[i] + ((orbits_GK_x[i]+orbits_GK_x[i+1])/2.0)*dt*stepskip*10**(-15)
#     GK_int_orbits_y[i+1]  = GK_int_orbits_y[i] + ((orbits_GK_y[i]+orbits_GK_y[i+1])/2.0)*dt*stepskip*10**(-15)
#     GK_int_orbits_z[i+1]  = GK_int_orbits_z[i] + ((orbits_GK_z[i]+orbits_GK_z[i+1])/2.0)*dt*stepskip*10**(-15)
#     pass

# GK_int_correct = (GK_int_correct_x+GK_int_correct_y+GK_int_correct_z)/3
# GK_int_orbits = (GK_int_orbits_x+GK_int_orbits_y+GK_int_orbits_z)/3

# time = np.arange(1,nmsdtime)*stepskip*dt*10**(-3)
# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_correct_x)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_int_x.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_orbits_x)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_int_x.png")
# plt.close()

# #figure detail

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_correct_y)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_int_y.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_orbits_y)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_int_y.png")
# plt.close()


# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_correct_z)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_int_z.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# plt.plot(time,GK_int_orbits_z)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_int_z.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)
# plt.plot(time,GK_int_correct,color="red")


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_correct_int.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)
# plt.plot(time,GK_int_orbits)


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_pred_int.png")
# plt.close()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(111)

# ax.yaxis.offsetText.set_fontsize(40)
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------

# ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)

# plt.plot(time,GK_int_orbits,color="blue")
# plt.plot(time,GK_int_correct,color = "red")


# plt.xlabel("time [ps]",fontsize = 30)
# plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

# # plt.legend(fontsize = 30)

# plt.minorticks_on()

# ax.tick_params(labelsize = 30, which = "both", direction = "in")
# plt.tight_layout()
# plt.show()

# plt.savefig(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/"+"GK_int_pred_and_correct.png")
# plt.close()



# D_PREDICTED = np.average(GK_int_orbits[int(0.6*nmsdtime):])

# info_ad = pd.DataFrame(data=[["D_pred_GK [m$^2$/s]",D_PREDICTED]],columns = columns2)
# info = pd.concat([info,info_ad])


# D_CORRECT = np.average(GK_int_correct[int(0.6*nmsdtime):])

# info_ad = pd.DataFrame(data=[["D_correct_GK [m$^2$/s]",D_CORRECT ]],columns = columns2)
# info = pd.concat([info,info_ad])


# info.to_csv(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/info3.txt",index = False)

# VACF_temp = []
# D_int_temp = []
# for i in time:
#     VACF_temp.append(i)
#     D_int_temp.append(i)
#     pass

# for i in correct_GK:
#     VACF_temp.append(i)
#     pass

# for i in orbits_GK:
#     VACF_temp.append(i)
#     pass

# for i in GK_int_correct:
#     D_int_temp.append(i)
#     pass

# for i in GK_int_orbits:
#     D_int_temp.append(i)
#     pass


# np.savetxt(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/VACF.txt",VACF_temp)
# np.savetxt(r"/home/s_tanaka/MD-GAN/test/seed"+str(seed)+"/D_int.txt",D_int_temp)