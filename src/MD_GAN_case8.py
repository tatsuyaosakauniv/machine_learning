import numpy as np
import scipy as sp
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import os
import time

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #GPUが使うメモリの指定．ない場合，一回のプログラム実行でGPUのメモリをすべて使う．
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def use_cpu():
    #CPU計算をしたい
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #環境変数を指定するだけ．
    return

def use_gpu():
    #GPU計算をしたい
    return

use_gpu()


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


####
#### pytorch風の実装を行う．
####

from matplotlib.ticker import ScalarFormatter

class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                useMathText=useMathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._order_of_mag

#Times new roman いつかは出来るようにしたいlinuxでのTimes New Romanでの描画.今はFontがないですって言われる．
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"]="stix"

#seed固定 多分完全にシード値の固定をして再現性を保証する事は出来ないと思われる．

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

#データ処理開始###################
seed_list = [1]
for seed in seed_list:

    set_seed(seed)

    #open check
    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/open_check.txt",[seed]) # seed番号を記録

    address = r"/home/s_tanaka/result/pat8" + "/"               #r"[ファイルが入ってるフォルダー名]"+"/"

    #Argon トラジェクトリデータ

    number_of_file = 10

    Ar_displacement_filename = "veloc_comp"+str(number_of_file)+".dat"                 #学習用ファイル
    data_name = address+Ar_displacement_filename

    Ar_DISPLACEMENT = np.loadtxt(data_name)

    stepskip = 2**( number_of_file -1)

    if(number_of_file==10):
        stepskip=15
        pass


    Ar_la_Const = 5.34

    nx_ar = 16
    ny_ar = 8
    nz_ar = 16

    controlled_T = 120

    dt = 2.0 #時間刻み [fs]
    fs = 1.0E-15

    ## アニメーションparam #元々やろうとしてたものの名残．使わないけど，point_mol_numとか一部使ってるのもあって消すに消せない．############

    ########################################################################################################################
    #データ前処理用の色々
    data_step = 100000 #MDのサンプルから取り出してくるデータ長
    use_step = 30000   #学習に使うデータ長

    Ar_molnum = nx_ar*ny_ar*nz_ar #MDの総分子数

    #!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
    columns2 = ["parameter","value"]
    info = pd.DataFrame(data = [["data_step",data_step]],columns= columns2)
    info_ad = pd.DataFrame(data=[["use_step",use_step]],columns = columns2)
    info = pd.concat([info,info_ad])
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    allmolnum = 1000

    dimension = 3  # 次元   熱流束の場合必要ない

    point_mol_num = 1000   #比較用の分子数

    info_ad = pd.DataFrame(data=[["use_mol_num",point_mol_num]],columns = columns2)
    info = pd.concat([info,info_ad])


    ###予測データとの比較用にMDデータの処理
    # correct_disp = np.zeros(shape =(Ar_molnum,dimension,data_step))   # 後で定義しているので、必要ない？

    #######################################
    """
    Fortranでのデータの出力形状を取り扱いやすいように成形する．

    Fortranの出力については，

    分子番号  x, y, z   (t=1)
    1        〇 〇 〇
    2        〇 〇 〇
    3        〇 〇 〇
    4        〇 〇 〇
    ...
    n-1      〇 〇 〇
    n        〇 〇 〇
                                        (.datファイル上にはこの空白はない)
    1        〇 〇 〇   (t=2)
    2        〇 〇 〇
    3        〇 〇 〇
    4        〇 〇 〇
    ...

    こんな感じになっている．

    これを
    [
    (分子1) ( t=1 t=2 t=3 t=4 .....                                )
    x        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]
    y        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]
    z        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]

    (分子2) ( t=1 t=2 t=3 t=4 .....                                )
    x        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]
    y        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]
    z        [〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇, 〇,...., 〇, 〇]
    ]
    こんな感じにする．
    """
    #######################################

    # for i in range(0,Ar_molnum):
    #     temp = np.zeros(shape = (data_step,dimension))      #データの仮の入れ物（[分子のトラジェクトリ,次元xyz]）
    #     for j in range(1,data_step+1):                      #このfor文で一個ずつデータを入れていく
    #         temp[j-1,:] = Ar_DISPLACEMENT[Ar_molnum*j+i,:dimension]#MDでのデータ出力のやり方上，ちょっときもい整理の仕方になる．
    #         pass
    #     temp = np.transpose(temp)                           #機械学習の出力形状に合わせる為に一回転置を挟む．
    #     correct_disp[i,:,:] = temp
    #     pass

    ###予測データとの比較用にMDデータの処理
    correct_disp = np.zeros(shape =(point_mol_num,dimension,data_step))

    for i in range(0,point_mol_num):
        temp = np.zeros(shape = (data_step,dimension))      #データの仮の入れ物（[分子のトラジェクトリ,次元xyz]）
        for j in range(1,data_step+1):                      #このfor文で一個ずつデータを入れていく
            temp[j-1,:] = Ar_DISPLACEMENT[Ar_molnum*j+i,:dimension]#MDでのデータ出力のやり方上，ちょっときもい整理の仕方になる．
            pass
        temp = np.transpose(temp)                           #機械学習の出力形状に合わせる為に一回転置を挟む．
        correct_disp[i,:,:] = temp
        pass
    #############################

    #Argon トラジェクトリデータ
    Ar_displacement_filename = "veloc_ar"+str(number_of_file)+".dat"                   #学習用ファイル
    data_name = address+Ar_displacement_filename

    Ar_DISPLACEMENT = np.loadtxt(data_name)

    #Ar_dispデータ取り出し
    Ar_disp = np.zeros(shape =(point_mol_num,dimension,use_step))

    #分子番号1~〇〇を取り出すのではなく，分子番号を無作為に指定するための配列
    rand_point_mol_num = [random.randint(0,point_mol_num-1) for i in range(point_mol_num)]

    print(Ar_DISPLACEMENT[1,0])
    Ar_DISPLACEMENT = np.array(Ar_DISPLACEMENT)

    ###学習用データの処理################
    for i in range(0,point_mol_num):
        temp = np.zeros(shape = (use_step,dimension)) #データの仮の入れ物（[分子のトラジェクトリ,次元xyz]）
        for j in range(1,use_step+1):                 #このfor文で一個ずつデータを入れていく
            temp[j-1,:] = Ar_DISPLACEMENT[allmolnum*j+rand_point_mol_num[i],:dimension]
            pass
        temp = np.transpose(temp)
        Ar_disp[i,:,:] = temp
        pass
    del Ar_DISPLACEMENT
    #####################################
    print(Ar_disp)
    print(np.shape(Ar_disp))

    # x, y, z それぞれで標準化している
    average_Ar_dispx = np.average(Ar_disp[:,0,:])
    std_Ar_dispx = np.std(Ar_disp[:,0,:])
    Ar_disp[:,0,:] = (Ar_disp[:,0,:]-average_Ar_dispx)/std_Ar_dispx
                    
    average_Ar_dispy = np.average(Ar_disp[:,1,:])
    std_Ar_dispy = np.std(Ar_disp[:,1,:])
    Ar_disp[:,1,:] = (Ar_disp[:,1,:]-average_Ar_dispy)/std_Ar_dispy
                    
    average_Ar_dispz = np.average(Ar_disp[:,2,:])
    std_Ar_dispz = np.std(Ar_disp[:,2,:])
    Ar_disp[:,2,:] = (Ar_disp[:,2,:]-average_Ar_dispz)/std_Ar_dispz

    info_ad = pd.DataFrame(data=[["x ave",average_Ar_dispx]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["x std",std_Ar_dispx]],columns = columns2)
    info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["y ave",average_Ar_dispy]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["y std",std_Ar_dispy]],columns = columns2)
    info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["z ave",average_Ar_dispz]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["z std",std_Ar_dispz]],columns = columns2)
    info = pd.concat([info,info_ad])


    ###################################################
    #tensorflow で取り扱えるようにデータ形状の前処理
    """
    train_dataに，入れれるだけのトラジェクトリサンプルを与える．

    sequence_length      : 一回の予測で生成されるデータ長
    data_length_per_atom : 一つの分子からとれるデータサンプル長 （学習用のデータ長/一回の予測のデータ長）
    iteration_all        : 勾配を更新する回数の総合
    trajectory_iteration_per_model : モデルを保存する勾配更新回数（iteration_all/trajectory_iteration_per_model が モデルの数）
    batch_size           : バッチ数
    all_data_NUM         : 今回のプログラム実行で必要となるデータ総数．
    """
    #Ar_disp = [分子番号,次元,トラジェクトリ]　<-----------------------------------------------------------------------------!!!!!!重要!!!!!!
    print(Ar_disp)
    Ar_disp = np.transpose(Ar_disp,(0,2,1))
    #Ar_disp = [分子番号,トラジェクトリ,次元]   <------------------------------------------------------------------------ 時間の入力がない？？
    print(np.shape(Ar_disp))

    sequence_length = 64

    data_length_per_atom = int(use_step/sequence_length)

    iteration_all = 150000+data_length_per_atom
    trajectory_iteration_per_model = 10000
    batch_size = 64

    all_data_NUM = iteration_all*batch_size

    print(all_data_NUM)

    info_ad = pd.DataFrame(data=[["sequence_length",sequence_length]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["iteration_all",iteration_all]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["trajectory_iteration_per_model",trajectory_iteration_per_model]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["batch_size",batch_size]],columns = columns2)
    info = pd.concat([info,info_ad])

    #データ処理終了##################################################

    #######################################################################################
    ################################# モデル訓練パート #####################################
    #######################################################################################
    # menas = 0
    # stds = 0.2
    random_uniform_inf = 0
    random_uniform_sup = 1.0
    menas2 = 0
    stds2 = 1.0

    dim = 3

    latent_dim = 32


    discriminator_extra_steps = 5
    latent_lr = 0.000002
    gen_lr = 0.000002
    disc_lr = 0.0000004

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to discriminator loss.
    def discriminator_loss(true_data, recon_data):
        return -(tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data))

    # Define the loss functions for the generator. 誤差のmaximaizeはtensorflowにはない．https://www.brainpad.co.jp/doors/contents/01_tech_2017-09-08-140000/
    def generator_loss(true_data,recon_data): 
        return tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data)

    def zgenerator_loss(true_data,recon_data):
        return tf.math.reduce_mean(recon_data) - tf.math.reduce_mean(true_data)



    #WGAN-GPではAdamを，WGANではRMSpropをなんて話を目にした．https://techblog.cccmkhd.co.jp/entry/2022/04/26/074557 いや，原著論文読めよ俺．
    zgenerator_optimizer = keras.optimizers.RMSprop(
        learning_rate = latent_lr
    )

    generator_optimizer = keras.optimizers.RMSprop(
        learning_rate = gen_lr
    )

    discriminator_optimizer = keras.optimizers.RMSprop(
        learning_rate = disc_lr
    )

    #!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
    info_ad = pd.DataFrame(data=[["latent_dims",latent_dim]],columns = columns2)
    info = pd.concat([info,info_ad])
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
    info_ad = pd.DataFrame(data=[["latent_lr",latent_lr]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["gen_lr",gen_lr]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["disc_lr",disc_lr]],columns = columns2)
    info = pd.concat([info,info_ad])
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    gen_array = []
    latent_gen_array = []

    ###############
    #自作の色々欄
    class Normalization(tf.keras.layers.Layer): #入力値をmean,stdとなるようにする．
        def __init__(self,axis):
            super(Normalization,self).__init__()
            self.axis = axis
        
        def call(self,inputs):
            return (inputs - tf.math.reduce_mean(inputs,axis=self.axis,keepsdims=True))/tf.math.reduce_std(inputs,axis=self.axis,keepsdims=True)


    # class Linear_tanh(tf.keras.layers.Layer):
    #     def __init__(self):
    #         super(Linear_tanh,self).__init__()
        
    #     def call(self,inputs):
    #         return inputs + tf.math.tanh(inputs)

    def Linear_tanh(z,alpha = 0.16):
        return alpha*z + tf.math.tanh(z)

    Normalize_axis = 1


    ###############################################学習済みモデル呼び出し時はここから「学習モデル呼び出し」までをコメントアウト
    ################################
    #### build latent generator ####
    ################################
    #この後このパラメータは使わない．（標準化処理はしているけど）
    mean_for_std = 0    #標準化 平均
    var_for_std = 1.0   #標準化 標準偏差


    #出力標準化処理用パラメータ ちなみに，#-- input1
    latent_inputs = tf.keras.Input(shape = (latent_dim,1))
    latent_layers = tf.keras.layers.Flatten()(latent_inputs)

    #-- input2
    random_inputs_latent = tf.keras.Input(shape = (latent_dim,1))
    random_latent_layers = tf.keras.layers.Flatten()(random_inputs_latent)

    #-- hidden layers
    la_concat_layer = tf.keras.layers.Concatenate(axis=1)([latent_layers,random_latent_layers])
    x1 = tf.keras.layers.Dense(units = 1024,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(la_concat_layer)
    x2 = tf.keras.layers.Dense(units = 1024,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x1)
    x3 = tf.keras.layers.Dense(units = 1024)(x2)

    output = tf.keras.layers.Dense(units = latent_dim)(x3)

    latent_gen = tf.keras.Model([latent_inputs,random_inputs_latent], output,name = "latent_generator")

    latent_gen.summary()

    #########################
    #### build generator ####
    #########################

    #-- input1
    AE_inputs = tf.keras.Input(shape = (latent_dim,1))
    inx = tf.keras.layers.Flatten()(AE_inputs)

    # mapping
    outx = tf.keras.layers.Dense(units = latent_dim,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(inx)
    x = tf.keras.layers.Concatenate(axis=1)([inx,outx])

    for _ in range(3):
        outx = tf.keras.layers.Dense(units = latent_dim,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x)
        x = tf.keras.layers.Concatenate(axis=1)([inx,outx])

    flat_AE = x

    #-- input2
    noise_inputs = tf.keras.Input(shape = (sequence_length,dim))#ガウスノイズ
    c_noise = tf.keras.layers.Conv1D(filters = 64, kernel_size = 1,strides = 1,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(noise_inputs)
    flat_noise_g = tf.keras.layers.Flatten()(c_noise)
    innx = tf.keras.layers.Dense(units = sequence_length*dim,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(flat_noise_g)
    # mapping
    nxout = tf.keras.layers.Dense(units = sequence_length*dim,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(innx)
    nx = tf.keras.layers.Concatenate(axis=1)([innx,nxout])

    for _ in range(3):
        nxout = tf.keras.layers.Dense(units = sequence_length*dim,activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(nx)
        nx = tf.keras.layers.Concatenate(axis=1)([innx,nxout])

    flat = nx

    #-- hidden layers
    concat_layer = tf.keras.layers.Concatenate(axis=1)([flat,flat_AE])
    x1 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(concat_layer)
    x2 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x1)
    x3 = tf.keras.layers.Dense(2048, activation = tf.keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal")(x2)

    #-- output
    decoded = tf.keras.layers.Dense(units = sequence_length*dim)(x3)

    generator = tf.keras.Model([noise_inputs,AE_inputs],decoded, name = "generator")

    generator.summary()


    #############################
    #### build discriminator ####
    #############################

    #-- input
    disc_inputs = keras.Input(shape = (sequence_length*3,dim))

    #-- hidden layers
    dc1 = layers.Conv1D(filters = 2048, kernel_size = sequence_length*3,strides = sequence_length*3,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
    flat = layers.Flatten()(dc1)


    #
    dc2 = layers.Conv1D(filters = 16, kernel_size = 1,strides = 1,activation = keras.layers.LeakyReLU(alpha = 0.3),kernel_initializer = "he_normal",padding = 'valid')(disc_inputs)
    flat2 = layers.Flatten()(dc2)

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

    書き方をpytorchに寄せる．つまるところdefine by Runで学習を行うという事．fitにデータを入れるやつはdefine and run．
    違いとしては
    予め計算のシステムを作り上げて，そこにデータを流し込むやり方→define and run
    データを生み出しながら正解と比較していくやり方→define by run


    MD-GANは恐らく，後者に特化した作り方．論文の書き方を見るに，開発者はpyTorchで書いてるはず．
    あちらはデフォルトでdefine by runしかできないが，Tensorflowはどちらもできる．（広く知れ渡ってるのはdefine and run）
    Tensorflowにある．
    Eager Executionというもの．これがdefine by runをtensorflowで可能にするインターフェース．
    ver1.5まではsess.run()とかの宣言が必要だったとか．その後ver2.0を出してシンプルになったとか．

    書き方はここと公式のリファレンスを見ながら
    https://wshinya.hatenablog.com/entry/2019/10/18/144025
    https://www.tensorflow.org/guide/function?hl=ja

    @tf.functionをつけるとgraphモード               (define and run)
    @tf.fuctionなし(コメントアウト)だとEagerモード   (define by run)

    実際のところ，Eagerモードは計算が遅い（気がする．）ので，
    二つの良いところ取りをするような訓練過程を構築しようと思う．

    1. 初期値生成（論文に準拠）．Snz(一様乱数 [0,1])を初期値として，分布緩和を導入．7~10回の繰り返し生成で分布がおよそ一定になるらしい．

    2. iteration回の勾配更新実行．(train_step関数呼び出し)

    2-a. 前回の学習もしくは1.で生成した潜在変数を受け取る．
    2-b. discriminatorの学習(WGANのリファレンスコードではgeneratorの5倍ほど学習回数を増やし，その代わり学習率を1/5にするみたいなことがあった)
    2-c. generatorとlatent_generatorの学習（今は同時の学習にしているが，このやり方にする前はどちらでも精度や挙動が変わらなかった．だったらこっちのが処理早いし．）
    2-d. 学習で生成した乱数をreturn(次の学習で使う)

    3. d_lossとかの表示の奴をリセットする．じゃないとメモリを圧迫する．パソコンが逼迫する．

    4. iterationが10000回（論文準拠）ごとにトラジェクトリを生成する（事が出来るモデルを保存する．）


    """


    train_d_loss = tf.keras.metrics.Mean()
    train_g_loss = tf.keras.metrics.Mean()
    train_gz_loss = tf.keras.metrics.Mean()

    @tf.function()
    def save_judge(train_sample,latent):    # 追加されてる　あんまりわからん

        #潜在変数生成
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent1 = latent_gen([latent_input,random_noise],training = False)
        
        latent_mean = tf.math.reduce_mean(latent1,axis=1,keepdims=True)
        std_mean = tf.math.reduce_std(latent1,axis=1,keepdims=True)
        
        latent1 = (latent1-latent_mean)/std_mean
        #潜在変数二つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent2 = latent_gen([latent1,random_noise],training = False)

        latent_mean = tf.math.reduce_mean(latent2,axis=1,keepdims=True)
        std_mean = tf.math.reduce_std(latent2,axis=1,keepdims=True)
        
        latent2 = (latent2-latent_mean)/std_mean

        #潜在変数三つ目
        random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
        latent3 = latent_gen([latent2,random_noise],training = False)

        latent_mean = tf.math.reduce_mean(latent3,axis=1,keepdims=True)
        std_mean = tf.math.reduce_std(latent3,axis=1,keepdims=True)
        
        latent3 = (latent3-latent_mean)/std_mean
        
        #データ生成
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################

        reconstruction = generator([noise_inputer,latent],training = False)
        
        reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
        
        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################

        reconstruction2 = generator([noise_inputer,latent2],training = False)

        reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])

        ########################################################

        #ノイズ入力準備#########################################
        noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
        #######################################################

        reconstruction3 = generator([noise_inputer,latent3],training = False)

        reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])

        ########################################################

        reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)

        #偽データの評価値生成
        fake_logits_temp = discriminator(reconstruction_sample,training = False)
        fake_logits = fake_logits_temp
        true_logits = discriminator(train_sample,training = False)

        g_loss = generator_loss(true_logits,fake_logits)
        return g_loss

    #訓練過程の定義
    ###############################################################################
    ###############################################################################
    ###############################################################################
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

    #GP_penaltyの実装
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

    def normalize_latent(latent):
        latent_mean = tf.math.reduce_mean(latent,axis=Normalize_axis,keepdims=True)
        std_mean = tf.math.reduce_std(latent,axis=Normalize_axis,keepdims=True)
        
        latent = (latent-latent_mean)/std_mean
        return latent

        
    @tf.function()
    def train_step(train_sample,latent_input):
        data = train_sample
        d_steps_add = discriminator_extra_steps
        gp_weight = 10
        ########################
        # discriminator学習開始 #
        ########################
        for l in range(d_steps_add):
            with tf.GradientTape() as tape :
                # 潜在変数部分が追加されてる

                #潜在変数生成
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent = latent_gen([latent_input,random_noise],training = False)
                
                latent = normalize_latent(latent)
                #潜在変数二つ目
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent2 = latent_gen([latent,random_noise],training = False)
                
                latent2 = normalize_latent(latent2)
                #潜在変数三つ目
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent3 = latent_gen([latent2,random_noise],training = False)

                latent3 = normalize_latent(latent3)

                #データ生成             reconstruction がなんか増えてる
                #ノイズ入力準備#########################################
                noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
                #######################################################

                reconstruction = generator([noise_inputer,latent],training = False)
                
                reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
                
                #ノイズ入力準備#########################################
                noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
                #######################################################

                reconstruction2 = generator([noise_inputer,latent2],training = False)

                reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])

                #ノイズ入力準備#########################################
                noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
                #######################################################

                reconstruction3 = generator([noise_inputer,latent3],training = False)

                reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])

                ########################################################

                reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)
                #偽データの評価値生成
                fake_logits_temp = discriminator(reconstruction_sample,training = True)
                fake_logits = fake_logits_temp
                
                true_logits = discriminator(data,training = True)
                d_cost = discriminator_loss(true_logits,fake_logits)
                gp =  gradient_penalty(data,reconstruction_sample,batch_size)
                d_cost = tf.cast(d_cost, tf.float64)
                d_loss = d_cost + gp*gp_weight
                
            d_gradient = tape.gradient(d_loss,discriminator.trainable_variables)

            discriminator_optimizer.apply_gradients(
                zip(d_gradient,discriminator.trainable_variables)
            )

        ########################################
        # latent generator学習開始 #
        ########################################
        with tf.GradientTape() as tape :
            #潜在変数生成
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent = latent_gen([latent_input,random_noise],training = True)
            
            latent = normalize_latent(latent)
            #潜在変数二つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent2 = latent_gen([latent,random_noise],training = True)

            latent2 = normalize_latent(latent2)
            #潜在変数三つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent3 = latent_gen([latent2,random_noise],training = True)

            latent3 = normalize_latent(latent3)
            
            #データ生成
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction = generator([noise_inputer,latent],training = False)
            
            reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
            
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction2 = generator([noise_inputer,latent2],training = False)

            reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])

            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction3 = generator([noise_inputer,latent3],training = False)

            reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])

            ########################################################

            reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)

            #偽データの評価値生成
            fake_logits_temp = discriminator(reconstruction_sample,training = False)
            fake_logits = fake_logits_temp
            true_logits = discriminator(data,training = False)

            gz_cost = zgenerator_loss(true_logits,fake_logits)

            gp =  gradient_penaltyG(data,reconstruction_sample,batch_size)
            gz_cost = tf.cast(gz_cost, tf.float64)
            gz_loss = gz_cost - gp*gp_weight
            
        genz_gradient = tape.gradient(gz_loss,latent_gen.trainable_variables)

        zgenerator_optimizer.apply_gradients(
            zip(genz_gradient,latent_gen.trainable_variables)
        )

        ########################################
        # generator学習開始 #
        ########################################
        with tf.GradientTape() as tape :
            #潜在変数生成
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent1 = latent_gen([latent_input,random_noise],training = False)
            
            latent_mean = tf.math.reduce_mean(latent1,axis=1,keepdims=True)
            std_mean = tf.math.reduce_std(latent1,axis=1,keepdims=True)
            
            latent1 = (latent1-latent_mean)/std_mean
            #潜在変数二つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent2 = latent_gen([latent1,random_noise],training = False)

            latent2 = normalize_latent(latent2)

            #潜在変数三つ目
            random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent3 = latent_gen([latent2,random_noise],training = False)

            latent3 = normalize_latent(latent3)

            #データ生成
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction = generator([noise_inputer,latent1],training = True)
            
            reconstruction = tf.reshape(reconstruction,[batch_size,sequence_length,dim])
            
            #ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction2 = generator([noise_inputer,latent2],training = True)

            reconstruction2 = tf.reshape(reconstruction2,[batch_size,sequence_length,dim])

            ##ノイズ入力準備#########################################
            noise_inputer = tf.random.normal([batch_size,sequence_length,dim],mean = menas2,stddev = stds2)
            #######################################################

            reconstruction3 = generator([noise_inputer,latent3],training = True)

            reconstruction3 = tf.reshape(reconstruction3,[batch_size,sequence_length,dim])

            ########################################################

            reconstruction_sample = tf.concat([reconstruction,reconstruction2,reconstruction3],axis = 1)

            #偽データの評価値生成
            fake_logits_temp = discriminator(reconstruction_sample,training = False)
            fake_logits = fake_logits_temp
            true_logits = discriminator(data,training = False)

            g_cost = zgenerator_loss(true_logits,fake_logits)

            gp =  gradient_penaltyG(data,reconstruction_sample,batch_size)
            g_cost = tf.cast(g_cost, tf.float64)
            g_loss = g_cost - gp*gp_weight
            
        gen_gradient = tape.gradient(g_loss,generator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gen_gradient,generator.trainable_variables)
        )

        train_d_loss.update_state(d_loss)
        train_g_loss.update_state(g_loss)
        train_gz_loss.update_state(gz_loss)

        return latent
    ##############################################################################################
    ################################################################################################
    ##################################################################################################

    gen_array = []
    latent_gen_array = []

    Early_merge = 1

    iteration_list = []
    gL_list = []
    latent_gL_list = []
    dL_list = []

    save_point = []
    save_iteration = []

    pass_level = 0.001

    info_ad = pd.DataFrame(data=[["Early_merge",Early_merge]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["pass_level",pass_level]],columns = columns2)
    info = pd.concat([info,info_ad])

    ########################
    #####   訓練実行   #####
    ########################

    counter_for_iteration = 0

    count_data_set = 0
    count_data_next = 0

    #-- training
    ## 1st
    for i in range(1,iteration_all+1):
        counter_for_iteration += 1
        if(count_data_set == 0):
            #初期値生成
            latent_initial = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            random_noise  =  tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

            latent = latent_gen([latent_initial,random_noise])

            latent = normalize_latent(latent)

            ########################################
            #分布緩和
            rand_int_for_initial = random.randrange(start = 12,stop = 15,step = 1)

            for iterations in range(rand_int_for_initial):#分布緩和
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent = latent_gen([latent,random_noise])

                latent = normalize_latent(latent)
                pass
            
            train_data = []

            indices = np.random.choice(point_mol_num,batch_size,replace=False)
            temp = []
            for j in range(data_length_per_atom - 2):
                temp.append(Ar_disp[indices,j*sequence_length:(j+3)*sequence_length,:])
            train_data = np.array(temp)
        
        input_data = train_data[count_data_set]

        latent = train_step(train_sample = input_data,latent_input = latent)
        count_data_set += 1

        #loss 出力
        average_d_loss = train_d_loss.result()
        average_g_loss = train_g_loss.result()
        average_gz_loss = train_gz_loss.result()

        #loss 画面表示
        print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}, gz_loss: {:4f}".format(i+1,average_d_loss,average_g_loss,average_gz_loss))

        #loss値のリセット
        train_d_loss.reset_states()
        train_g_loss.reset_states()
        train_gz_loss.reset_states()

        #学習曲線のリストメイク
        iteration_list.append(counter_for_iteration)
        gL_list.append(average_g_loss)
        latent_gL_list.append(average_gz_loss)
        dL_list .append(average_d_loss)
        if(count_data_set == data_length_per_atom-2):
            count_data_set = 0
            count_data_next += 1
            pass
        pass


    count_data_set = 0
    count_data_next = 0
    ## 2nd
    for i in range(1,iteration_all+1):
        counter_for_iteration += 1
        if(count_data_set == 0):
            #初期値生成
            latent_initial = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            random_noise  =  tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

            latent = latent_gen([latent_initial,random_noise])

            latent = normalize_latent(latent)

            ########################################
            #分布緩和
            rand_int_for_initial = random.randrange(start = 12,stop = 15,step = 1)

            for iterations in range(rand_int_for_initial):#分布緩和
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent = latent_gen([latent,random_noise])

                latent = normalize_latent(latent)
                pass

            train_data = []

            indices = np.random.choice(point_mol_num,batch_size,replace=False)
            temp = []
            for j in range(data_length_per_atom - 2):
                temp.append(Ar_disp[indices,j*sequence_length:(j+3)*sequence_length,:])
            train_data = np.array(temp)
        
        input_data = train_data[count_data_set]

        latent = train_step(train_sample = input_data,latent_input = latent)
        count_data_set += 1

        #loss 出力
        average_d_loss = train_d_loss.result()
        average_g_loss = train_g_loss.result()
        average_gz_loss = train_gz_loss.result()

        print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}, gz_loss: {:4f}".format(i+1,average_d_loss,average_g_loss,average_gz_loss))

        train_d_loss.reset_states()
        train_g_loss.reset_states()
        train_gz_loss.reset_states()

        iteration_list.append(counter_for_iteration)
        gL_list.append(average_g_loss)
        latent_gL_list.append(average_gz_loss)
        dL_list .append(average_d_loss)
        if(count_data_set == data_length_per_atom-2):
            count_data_set = 0
            count_data_next += 1
            pass
        pass

    count_data_set = 0
    count_data_next = 0
    ## 3rd
    for i in range(1,iteration_all+1):
        counter_for_iteration += 1
        if(count_data_set == 0):
            #初期値生成
            latent_initial = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            random_noise  =  tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

            latent = latent_gen([latent_initial,random_noise])

            latent = normalize_latent(latent)

            ########################################
            #分布緩和
            rand_int_for_initial = random.randrange(start = 12,stop = 15,step = 1)

            for iterations in range(rand_int_for_initial):#分布緩和
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent = latent_gen([latent,random_noise])

                latent = normalize_latent(latent)
                pass

            train_data = []

            indices = np.random.choice(point_mol_num,batch_size,replace=False)
            temp = []
            for j in range(data_length_per_atom - 2):
                temp.append(Ar_disp[indices,j*sequence_length:(j+3)*sequence_length,:])
            train_data = np.array(temp)
        
        input_data = train_data[count_data_set]

        latent = train_step(train_sample = input_data,latent_input = latent)
        count_data_set += 1

        #loss 出力
        average_d_loss = train_d_loss.result()
        average_g_loss = train_g_loss.result()
        average_gz_loss = train_gz_loss.result()

        print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}, gz_loss: {:4f}".format(i+1,average_d_loss,average_g_loss,average_gz_loss))

        train_d_loss.reset_states()
        train_g_loss.reset_states()
        train_gz_loss.reset_states()

        iteration_list.append(counter_for_iteration)
        gL_list.append(average_g_loss)
        latent_gL_list.append(average_gz_loss)
        dL_list .append(average_d_loss)
        if(count_data_set == data_length_per_atom-2):
            count_data_set = 0
            count_data_next += 1
            pass
        pass

    #-- training

    count_data_set = 0
    count_data_next = 0

    save_count = 1
    counter = 0
    time_start = time.time()
    for i in range(1,iteration_all*3+1):
        counter_for_iteration += 1
        # if(save_count == 31):
        #     break

        if(count_data_set == 0):
            #初期値生成
            latent_initial = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
            random_noise  =  tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

            latent = latent_gen([latent_initial,random_noise])

            latent = normalize_latent(latent)

            ########################################
            #分布緩和
            rand_int_for_initial = random.randrange(start = 12,stop = 15,step = 1)

            for iterations in range(rand_int_for_initial):#分布緩和
                random_noise = tf.random.uniform([batch_size,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)
                latent = latent_gen([latent,random_noise])

                latent = normalize_latent(latent)
                pass
            train_data = []

            indices = np.random.choice(point_mol_num,batch_size,replace=False)
            temp = []
            for j in range(data_length_per_atom - 2):
                temp.append(Ar_disp[indices,j*sequence_length:(j+3)*sequence_length,:])
            train_data = np.array(temp)
        
        input_data = train_data[count_data_set]


        # save_g_loss = save_judge(train_sample = input_data,latent = latent)

        # if(np.abs(save_g_loss) < pass_level):
        #     counter+=1
        #     if(counter == 1):
        #         save_gen = generator
        #         save_latentgen = latent_gen
        #         pass

        #     if(counter == Early_merge):
        #         save_gen.save(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/252/models/"+"MD_GAN_gen_"+str(save_count)+".h5")
        #         save_latentgen.save(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/MD_GAN_zgen_"+str(save_count)+".h5")
        #         gen_array.append(save_gen)
        #         latent_gen_array.append(save_latentgen)
        #         counter = 0
        #         save_count +=1

        #         save_point.append(0)
        #         save_iteration.append(i-Early_merge)
        #         pass
        # else:
        #     counter = 0
        #     pass

        if(np.mod(i,5000) == 0):
            generator.save(r"/home/s_tanaka/MD-GAN/models/case8/seed"+str(seed)+"/MD_GAN_gen_"+str(save_count)+".h5")
            latent_gen.save(r"/home/s_tanaka/MD-GAN/models/case8/seed"+str(seed)+"/MD_GAN_zgen_"+str(save_count)+".h5")
            gen_array.append(generator)
            latent_gen_array.append(latent_gen)
            save_count +=1
            pass

        latent = train_step(train_sample = input_data,latent_input = latent)
        count_data_set += 1

        #loss 出力
        average_d_loss = train_d_loss.result()
        average_g_loss = train_g_loss.result()
        average_gz_loss = train_gz_loss.result()

        print("iteration: {:}, d_loss: {:4f}, g_loss: {:4f}, gz_loss: {:4f}".format(i+1,average_d_loss,average_g_loss,average_gz_loss))

        train_d_loss.reset_states()
        train_g_loss.reset_states()
        train_gz_loss.reset_states()

        iteration_list.append(counter_for_iteration)
        gL_list.append(average_g_loss)
        latent_gL_list.append(average_gz_loss)
        dL_list .append(average_d_loss)
        if(count_data_set == data_length_per_atom-2):
            count_data_set = 0
            count_data_next += 1
            pass


    ############################学習曲線

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #plot
    ax.plot(iteration_list,dL_list,color = "red",label  = "discriminator Loss")
    ax.plot(iteration_list,gL_list,color = "blue",label  = "generator Loss")
    ax.plot(iteration_list,latent_gL_list,color = "green",label  = "latent generator Loss")

    ax.scatter(save_iteration,save_point,color = "black",marker = "*")



    ax.set_xlabel("iteration",fontsize = 30)
    ax.set_ylabel("Loss",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/trainin_proceed.png")
    plt.close()

    info_ad = pd.DataFrame(data=[["passed model",len(gen_array)]],columns = columns2)
    info = pd.concat([info,info_ad])

    ######################################################################################
    ############################# 学習済みモデル呼び出し ##################################
    ######################################################################################
    # gen_array = []
    # latent_gen_array = []

    # for i in range(1,2):

    #     #load generator
    #     generator = keras.models.load_model(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/models/"+"MD_GAN_gen_"+str(i)+".h5",compile = False)

    #     #load latetn_generator

    #     latent_gen = keras.models.load_model(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/models/"+"MD_GAN_zgen_"+str(i)+".h5",compile = False)
        
        
    #     gen_array.append(generator)
    #     latent_gen_array.append(latent_gen)
    #     pass

    # print("latent_gen_array :",len(latent_gen_array))
    # print("gen_array :",len(gen_array))
    #######################################################################################
    #######################################################################################
    #######################################################################################

    t = 2.0 #時間刻み [fs]


    #分布緩和処理関数

    def distribution_relax(latent_dimension,random_uniform_inf,random_uniform_sup,seed_value):
        #分布緩和の繰り返し回数．先行研究には ”MD-GAN with multi-particle input: the machine learning of long-time molecular behavior from short-time MD data✞"
        #7~10回からランダムに決めるようにするといい
        #って書いてあった．どすこい．
        rand_int_for_initial = random.randrange(start = 12,stop = 15,step = 1)

        latent_ini = tf.random.uniform([prediction_times,latent_dimension],minval = random_uniform_inf,maxval = random_uniform_sup)   #初期分布
        random_noise = tf.random.uniform([prediction_times,latent_dimension],minval = random_uniform_inf,maxval = random_uniform_sup) #入力乱数

        #生成
        latent = latent_gen.predict([latent_ini,random_noise])

        latent = normalize_latent(latent)
        

        for iterations in range(rand_int_for_initial):#分布緩和 
            random_noise = tf.random.uniform([prediction_times,latent_dimension],minval = random_uniform_inf,maxval = random_uniform_sup)
            latent = latent_gen.predict([latent,random_noise])

            latent = normalize_latent(latent)
            pass

        return latent

    ############################################################
    ####################### サンプル生成 ########################
    ############################################################
    prediction_times = 32    #サンプルの生成数（この数だけdata_stepのデータ長のトラジェクトリを生成する．）

    data_num = int(data_step/(sequence_length)) #繰り返し予測回数

    info_ad = pd.DataFrame(data=[["data_num",data_num]],columns = columns2)

    info = info = pd.concat([info,info_ad])


    generator = gen_array[0]
    latent_gen = latent_gen_array[0]    

    #分布緩和呼び出し
    latent_ini = distribution_relax(latent_dimension = latent_dim,random_uniform_inf = random_uniform_inf,random_uniform_sup = random_uniform_sup,seed_value = 0)
    latent_ini_input = latent_ini

    #ガウス分布乱数呼び出し
    noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)


    reconstruction_ini = generator.predict([noise_inputer,latent_ini_input])

    reconstruction_ini = tf.reshape(reconstruction_ini,[prediction_times,sequence_length,dim])

    latent_per_onemol = tf.expand_dims(latent_ini,axis =1)
    orbit_per_onemol = reconstruction_ini

    latent = latent_ini

    for j in range(data_num):
        #潜在変数生成###############################
        random_noise = tf.random.uniform([prediction_times,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

        latent = latent_gen.predict([latent,random_noise])

        latent = normalize_latent(latent)
        
        ###########################################

        latent_input = latent

        #サンプリング生成
        #ノイズ入力準備####################################
        noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)
        ##################################################

        reconstruction = generator.predict([noise_inputer,latent_input])

        reconstruction = tf.reshape(reconstruction,[prediction_times,sequence_length,dim])

        #データ追加#######################################
        save_latent = tf.expand_dims(latent,axis=1)
        latent_per_onemol = np.concatenate([latent_per_onemol,save_latent],axis = 1)
        orbit_per_onemol = np.concatenate([orbit_per_onemol,reconstruction],axis = 1)
        pass

    orbits = orbit_per_onemol
    latents = latent_per_onemol



    for i in range(1,len(gen_array)):
        generator = gen_array[i]
        latent_gen = latent_gen_array[i]    
        
        #分布緩和呼び出し
        latent_ini = distribution_relax(latent_dimension = latent_dim,random_uniform_inf = random_uniform_inf,random_uniform_sup = random_uniform_sup,seed_value = i)
        latent_ini_input = latent_ini

        #ガウス分布乱数呼び出し
        noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)

        reconstruction_ini = generator.predict([noise_inputer,latent_ini_input])

        reconstruction_ini = tf.reshape(reconstruction_ini,[prediction_times,sequence_length,dim])

        latent_per_onemol = tf.expand_dims(latent_ini,axis =1)
        orbit_per_onemol = reconstruction_ini

        latent = latent_ini

        for j in range(data_num):
            #潜在変数生成###############################
            random_noise = tf.random.uniform([prediction_times,latent_dim],minval = random_uniform_inf,maxval = random_uniform_sup)

            latent = latent_gen.predict([latent,random_noise])
            latent = normalize_latent(latent)
            
            ###########################################

            latent_input = latent

            #サンプリング生成
            #ノイズ入力準備####################################
            noise_inputer = tf.random.normal([prediction_times,sequence_length,dim],mean = menas2,stddev = stds2)
            ##################################################

            reconstruction = generator.predict([noise_inputer,latent_input])

            reconstruction = tf.reshape(reconstruction,[prediction_times,sequence_length,dim])

            #データ追加#######################################
            save_latent = tf.expand_dims(latent,axis=1)
            latent_per_onemol = np.concatenate([latent_per_onemol,save_latent],axis = 1)
            orbit_per_onemol = np.concatenate([orbit_per_onemol,reconstruction],axis = 1)
            pass

        orbits = np.concatenate([orbits,orbit_per_onemol],axis=0)
        latents = np.concatenate([latents,latent_per_onemol],axis=0)
        pass


    orbits = np.array(orbits)

    info_ad = pd.DataFrame(data=[["orbits NUM",np.shape(orbits)[0]]],columns = columns2)

    info = info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["x ave gened",np.average(orbits[:,:,0])]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["x std gened",np.std(orbits[:,:,0])]],columns = columns2)
    info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["y ave gened",np.average(orbits[:,:,1])]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["y std gened",np.std(orbits[:,:,1])]],columns = columns2)
    info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["z ave gened",np.average(orbits[:,:,2])]],columns = columns2)
    info = pd.concat([info,info_ad])
    info_ad = pd.DataFrame(data=[["z std gened",np.std(orbits[:,:,2])]],columns = columns2)
    info = pd.concat([info,info_ad])


    orbits[:,:,0] = (orbits[:,:,0]-np.average(orbits[:,:,0]))/np.std(orbits[:,:,0])
    orbits[:,:,0] = orbits[:,:,0]*std_Ar_dispx + average_Ar_dispx 

    orbits[:,:,1] = (orbits[:,:,1]-np.average(orbits[:,:,1]))/np.std(orbits[:,:,1])
    orbits[:,:,1] = orbits[:,:,1]*std_Ar_dispy + average_Ar_dispy 

    orbits[:,:,2] = (orbits[:,:,2]-np.average(orbits[:,:,2]))/np.std(orbits[:,:,2])
    orbits[:,:,2] = orbits[:,:,2]*std_Ar_dispz + average_Ar_dispz

    # orbits[:,:,1] = orbits[:,:,1]*std_Ar_dispy + average_Ar_dispy
    # y_ave = np.average(orbits[:,:,1].reshape(-1))
    # orbits[:,:,1] = orbits[:,:,1] - y_ave
    # orbits[:,:,1] = orbits[:,:,1] + average_Ar_dispy

    # orbits[:,:,2] = orbits[:,:,2]*std_Ar_dispz + average_Ar_dispz
    # z_ave = np.average(orbits[:,:,2].reshape(-1))
    # orbits[:,:,2] = orbits[:,:,2] - z_ave
    # orbits[:,:,2] = orbits[:,:,2] + average_Ar_dispz
    """
    orbits[分子番号,余計な項,分子の軌跡,xyz]
    ・分子番号 ; 分子の番号．
    ・余計な項 ; 生成モデルに入力する際にCNNを使うとき，(200,)は受け取れない．CNN入力層の入力サイズは(200,1)の形状を取っている必要がある．その名残
    ・分子の軌跡 ; 予測したトラジェクトリ
    ・xyz ; 訓練集合と同次元データの，xyz次元
    """

    orbits_x = orbits[:,:,0] #最後の0はx軸を表す．1はy軸方向で，2はz軸になる．
    orbits_y = orbits[:,:,1]
    orbits_z = orbits[:,:,2]


    print("prediction finished")


    print("shape",np.shape(orbits_x))
    print("correct_shape",np.shape(correct_disp)) #温度の計算する前にcorrect_dispのaxisを変えているので注意．この行時点では，[分子番号，xyz,速度の値配列（トラジェクトリ）]になっている．


    boundary_for_graph = use_step
    show_step = 3000

    ########以下，図示用の処理
    rand_array_fig = [random.randint(0,np.shape(orbits)[0]-1) for i in range(64)]

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        time_step = np.arange(1,np.shape(orbits_x[i])[0]+1)
        ax.plot(time_step,orbits_x[i])
        pass

    for i in range(64):
        time_step = np.arange(1,np.shape(correct_disp)[2]+1)
        ax.plot(time_step,correct_disp[i,0,:],color = "red")
        pass


    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+Ar_displacement_filename[:-4]+".png")
    plt.close()
    
    ###############
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(orbits_x[i]))
        time_step = np.arange(1,np.shape(orbits_x[i])[0]+1)
        ax.plot(time_step,orbits_x[i])
        pass

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/only_"+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #######

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    for i in range(64):
        time_step = np.arange(1,np.shape(correct_disp)[2]+1)
        ax.plot(time_step,correct_disp[i,0,:])
        pass

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/correct_"+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####detail
    #figure detail
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(orbits_x[i,10000:20000]))
        ax.plot(time_step[10000:20000],orbits_x[i,10000:20000])
        pass

    for i in range(64):
        print(np.shape(correct_disp[i,0,10000:20000]))
        ax.plot(time_step[10000:20000],correct_disp[i,0,10000:20000],color = "red")

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/10000-20000"+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####
    #figure detail
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #prediction displacement------------------------
    for i in range(1):
        print(np.shape(orbits_x[i,10000:20000]))
        ax.plot(time_step[10000:20000],orbits_x[i,10000:20000])
        pass

    for i in range(1):
        print(np.shape(correct_disp[i,0,10000:20000]))
        ax.plot(time_step[10000:20000],correct_disp[i,0,10000:20000],color = "red")

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/1atom from10000_beween"+str(10000)+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####
    #figure detail
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #prediction displacement------------------------
    for i in range(1):
        print(np.shape(orbits_x[i,10000:11000]))
        ax.plot(time_step[10000:10000+show_step],orbits_x[i,10000:10000+show_step])
        pass

    for i in range(1):
        print(np.shape(correct_disp[i,0,10000:10000+show_step]))
        ax.plot(time_step[10000:10000+show_step],correct_disp[i,0,10000:10000+show_step],color = "red")

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/1atom from10000_between"+str(show_step)+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####
    #####
    #figure detail
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #prediction displacement------------------------
    for i in range(1):
        print(np.shape(orbits_x[i,10000:11000]))
        ax.plot(time_step[10000:10000+show_step],orbits_x[i,10000:10000+show_step])
        pass
    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/1atom pred"+str(show_step)+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####
        #####
    #figure detail
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #prediction displacement------------------------
    for i in range(1):
        print(np.shape(correct_disp[i,0,10000:10000+show_step]))
        ax.plot(time_step[10000:10000+show_step],correct_disp[i,0,10000:10000+show_step],color = "red")
        pass

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)
 
    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/1atom correct"+str(show_step)+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####
    ##################################

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(orbits_x[i,0:1000]))
        ax.plot(time_step[0:0+show_step],orbits_x[i,0:0+show_step])
        pass

    for i in range(64):
        print(np.shape(correct_disp[i,0:1000]))
        ax.plot(time_step[0:0+show_step],correct_disp[i,0,0:0+show_step],color = "red")

    ax.set_xlabel("step",fontsize = 30)
    ax.set_ylabel("velocity",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/from0_between"+str(show_step)+Ar_displacement_filename[:-4]+".png")
    plt.close()
    #####


    #####
    ###########################
    #### 温度の時間遷移計算 ####
    ###########################

    info_ad = pd.DataFrame(data=[["orbits shape",np.shape(orbits)]],columns = columns2)

    info = info = pd.concat([info,info_ad])

    info_ad = pd.DataFrame(data=[["correct_disp shape",np.shape(correct_disp)]],columns = columns2)

    info = info = pd.concat([info,info_ad])



    M_ar = 39.948*10**(-3)
    Na = 6.022*10**23   #アボガドロ数
    kb = 1.3806662*10**(-23) #ボルツマン定数
    mass_ar = 10**26*M_ar/Na

    Temperature = np.zeros(np.shape(orbits)[1])

    for i in range(np.shape(orbits)[0]):
        velx = orbits[i,:,0]
        vely = orbits[i,:,1]
        velz = orbits[i,:,2]

        u_kine = (velx*velx + vely*vely + velz*velz)*0.5*mass_ar
        print(np.shape(u_kine))
        print(np.shape(Temperature))
        Temperature += u_kine
        pass

    Temperature = 2*Temperature/(3*kb*np.shape(orbits)[0])/10**16


    correct_disp = np.transpose(correct_disp,(0,2,1))##############################################################ここでcorrect_dispの転置をしているので注意######################correct_disp[分子番号,xyz,分子の軌跡]→correct_disp[分子番号,分子の軌跡,xyz]

    Temperature_correct = np.zeros(np.shape(correct_disp)[1])

    for i in range(np.shape(correct_disp)[0]):
        velx = correct_disp[i,:,0]
        vely = correct_disp[i,:,1]
        velz = correct_disp[i,:,2]

        u_kine = (velx*velx + vely*vely + velz*velz)*0.5*mass_ar

        print(np.shape(velx))
        print(np.shape(vely))
        print(np.shape(u_kine))
        print(np.shape(Temperature_correct))
        Temperature_correct += u_kine
        pass

    Temperature_correct = 2*Temperature_correct/(3*kb*np.shape(correct_disp)[0])/10**16
    print("tempcorrect")
    print(np.shape(Temperature_correct))



    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/pred_temp.dat",Temperature)
    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/correct_temp.dat",Temperature_correct)

    #_######################################

    #figure show
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))


    #figure detail

    time_step = np.arange(len(Temperature))
    ax.plot(time_step,Temperature,label = "predict temp",color = "#6495ED")


    time_step = np.arange(len(Temperature_correct))
    ax.plot(time_step,Temperature_correct,label = "true temp",color = "#EF4123")


    ax.set_xlabel("step",fontsize = 50)
    ax.set_ylabel("Temperature K",fontsize = 50)

    ax.legend(fontsize = 45)

    ax.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Temperature.png")
    plt.close()
    #figure show
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ################################################detail
    #figure detail

    time_step = np.arange(len(Temperature[0:1000]))
    ax.plot(time_step,Temperature[0:1000],label = "predict temp",color = "#6495ED")


    time_step = np.arange(len(Temperature_correct[0:1000]))
    ax.plot(time_step,Temperature_correct[0:1000],label = "true temp",color = "#EF4123")


    ax.set_xlabel("step",fontsize = 50)
    ax.set_ylabel("Temperature K",fontsize = 50)

    ax.legend(fontsize = 45)

    ax.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/detailed-Temperature.png")
    plt.close()

    ############################
    #### displacementの計算 ####
    ############################
    """
    シンプルである．ただ単純にorbitsを足していくだけ．
    サンプリングデータについて，元々は5step毎の抽出としている．
    MDは2[fs]ごとの更新なので，1ステップは10[fs]に相当する．

    MDでのサンプリングについては，5step分の平均速度値を.datに保存するようにしている．つまり，速度値×5stepした値は実際の変位と同じ値になる．
    これで計算を行っていく．

    作業用メモ
    orbits(現array)[分子番号，分子の軌跡，xyz]
    correct_disp[分子番号，分子の軌跡，xyz] correct_dispは温度を計算する前に転置しているので，取り扱いに注意．
    """


    length = np.shape(correct_disp)[1]
    x_disp_temp = np.zeros((np.shape(orbits)[0],length))
    y_disp_temp = np.zeros((np.shape(orbits)[0],length))
    z_disp_temp = np.zeros((np.shape(orbits)[0],length))
    temp_x = np.zeros((np.shape(orbits)[0]))
    temp_y = np.zeros((np.shape(orbits)[0]))
    temp_z = np.zeros((np.shape(orbits)[0]))

    for i in range(length):
        temp_x += orbits[:,i,0]*stepskip*dt
        temp_y += orbits[:,i,1]*stepskip*dt
        temp_z += orbits[:,i,2]*stepskip*dt

        x_disp_temp[:,i] = temp_x
        y_disp_temp[:,i] = temp_y
        z_disp_temp[:,i] = temp_z

        pass

    x_displacement = x_disp_temp
    y_displacement = y_disp_temp
    z_displacement = z_disp_temp

    x_cor_disp_temp = np.zeros((np.shape(correct_disp)[0],length))
    y_cor_disp_temp = np.zeros((np.shape(correct_disp)[0],length))
    z_cor_disp_temp = np.zeros((np.shape(correct_disp)[0],length))
    temp_cor_x = np.zeros((np.shape(correct_disp)[0]))
    temp_cor_y = np.zeros((np.shape(correct_disp)[0]))
    temp_cor_z = np.zeros((np.shape(correct_disp)[0]))

    for i in range(length):
        temp_cor_x += correct_disp[:,i,0]*stepskip*dt
        temp_cor_y += correct_disp[:,i,1]*stepskip*dt
        temp_cor_z += correct_disp[:,i,2]*stepskip*dt

        x_cor_disp_temp[:,i] = temp_cor_x
        y_cor_disp_temp[:,i] = temp_cor_y
        z_cor_disp_temp[:,i] = temp_cor_z
        pass

    x_correct_displacement = x_cor_disp_temp
    y_correct_displacement = y_cor_disp_temp
    z_correct_displacement = z_cor_disp_temp


    x_displacement = np.array(x_displacement)
    y_displacement = np.array(y_displacement)
    z_displacement = np.array(z_displacement)

    x_correct_displacement = np.array(x_correct_displacement)
    y_correct_displacement = np.array(y_correct_displacement)
    z_correct_displacement = np.array(z_correct_displacement)
    ###################
    #### MSDの計算 ####

    MSD_x = np.average(np.square(x_displacement),axis=0)
    MSD_y = np.average(np.square(y_displacement),axis=0)
    MSD_z = np.average(np.square(z_displacement),axis=0)

    MSD = (MSD_x+MSD_y+MSD_z)/3

    MSD_correct_x = np.average(np.square(x_correct_displacement),axis=0)
    MSD_correct_y = np.average(np.square(y_correct_displacement),axis=0)
    MSD_correct_z = np.average(np.square(z_correct_displacement),axis=0)

    MSD_correct = (MSD_correct_x+MSD_correct_y+MSD_correct_z)/3

    dmsdlog_correct = MSD_correct

    dmsdlogx_correct = MSD_correct_x
    dmsdlogy_correct = MSD_correct_y
    dmsdlogz_correct = MSD_correct_z

    dmsdlog_orbits = MSD

    dmsdlogx_orbits = MSD_x
    dmsdlogy_orbits = MSD_y
    dmsdlogz_orbits = MSD_z



    #10E-20表記にしてたんだけど，これはどうやら10×10**(-20)として扱われるらしい．
    msd_x_log_correct = dmsdlogx_correct*10**(-20)
    msd_y_log_correct = dmsdlogy_correct*10**(-20)
    msd_z_log_correct = dmsdlogz_correct*10**(-20)

    msd_log_correct = dmsdlog_correct*10**(-20)

    MSD_correct = msd_log_correct

    msd_x_log_orbits = dmsdlogx_orbits*10**(-20)
    msd_y_log_orbits = dmsdlogy_orbits*10**(-20)
    msd_z_log_orbits = dmsdlogz_orbits*10**(-20)

    msd_log_orbits = dmsdlog_orbits*10**(-20)

    MSD_orbits = msd_log_orbits

    time_correct = np.arange(1,len(MSD_correct)+1)*dt*10**(-3)*stepskip
    time_orbits  = np.arange(1,len(MSD)+1)*dt*10**(-3)*stepskip


    #################
    ####displacement 図描用にオーダー変更 Å → nm
    x_displacement = x_displacement*10**(-1) #[Å]→[nm]
    y_displacement = y_displacement*10**(-1)
    z_displacement = z_displacement*10**(-1)

    x_correct_displacement = x_correct_displacement*10**(-1)
    y_correct_displacement = y_correct_displacement*10**(-1)
    z_correct_displacement = z_correct_displacement*10**(-1)


    #x_displacement[分子番号，分子の変位] 他同

    time_step = np.arange(len(x_displacement[0,:]))*stepskip*dt*10**(-3) #×10 [fs] →×10D-3 [ps] 先行研究に単位を合わせる．



    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step,x_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("x displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/x-displacement.png")
    plt.close()
    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(y_displacement[i,:]))
        ax.plot(time_step,y_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("y displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/y-displacement.png")
    plt.close()
    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(z_displacement[i,:]))
        ax.plot(time_step,z_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("z displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/z-displacement.png")
    plt.close()
    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in range(64):
        time_step = np.arange(len(x_correct_displacement[0,:]))*stepskip*dt*10**(-3)
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step,x_correct_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("x displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/x-displacement_original.png")
    plt.close()
    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in range(64):
        time_step = np.arange(len(y_correct_displacement[0,:]))*stepskip*dt*10**(-3)
        print(np.shape(y_displacement[i,:]))
        ax.plot(time_step,y_correct_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("y displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/y-displacement_original.png")
    plt.close()
    ################################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in range(64):
        time_step = np.arange(len(z_correct_displacement[0,:]))*stepskip*dt*10**(-3)
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step,z_correct_displacement[i,:])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("z displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/z-displacement_original.png")
    plt.close()
    ##########################################細かく見ていく
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],x_displacement[i,0:0+show_step])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("x displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 x_displacement_pred.png")
    plt.close()
    ##########################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(y_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],y_displacement[i,0:0+show_step])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("y displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 y_displacement_pred.png")
    plt.close()
    ##########################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(z_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],z_displacement[i,0:0+show_step])
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("z displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 z_displacement_pred.png")
    plt.close()
    ##########################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],x_displacement[i,0:0+show_step])
        pass

    for i in range(64):
        print(np.shape(x_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],x_correct_displacement[i,0:0+show_step],color = "red")
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("x displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 x_displacement.png")
    plt.close()
    ##########################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(y_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],y_displacement[i,0:0+show_step])
        pass

    for i in range(64):
        print(np.shape(y_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],y_correct_displacement[i,0:0+show_step],color = "red")
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("y displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 y_displacement.png")
    plt.close()
    ##########################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #prediction displacement------------------------
    for i in rand_array_fig:
        print(np.shape(z_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],z_displacement[i,0:0+show_step])
        pass

    for i in range(64):
        print(np.shape(z_displacement[i,:]))
        ax.plot(time_step[0:0+show_step],z_correct_displacement[i,0:0+show_step],color = "red")
        pass

    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("z displacement [nm]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/0-2000 z_displacement.png")
    plt.close()


    #################################################


    time_step = np.arange(len(x_displacement[0,:]))*stepskip*dt*10**(-3) #×10 [fs] →×10D-3 [ps] 先行研究に単位を合わせる


    print(msd_log_correct)

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_correct,msd_log_correct,color = "red")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_correct.png")
    plt.close()
    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_correct,msd_x_log_correct,color = "red")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD x [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x_correct.png")
    plt.close()

    n = np.shape(time_correct)[0]
    print(n)

    time = time_correct*10**-12

    a = 1/2*(n*np.sum(time*MSD_correct)-np.sum(time)*np.sum(MSD_correct))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(MSD_correct)-np.dot(time,MSD_correct)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_md [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_orbits,msd_log_orbits,"--",color = "blue")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_predicted.png")
    plt.close()
    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_orbits,msd_x_log_orbits,"--",color = "blue")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD x [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x_predicted.png")
    plt.close()

    ############################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_orbits,msd_y_log_orbits,"--",color = "blue")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD y [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_y_predicted.png")
    plt.close()

    ############################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.plot(time_orbits,msd_z_log_orbits,"--",color = "blue")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD z [m$^2$]",fontsize = 30)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_z_predicted.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------
    ax.axvspan(0,use_step*dt*stepskip*10**-3,color = "coral",alpha = 0.5)

    ax.plot(time_correct,msd_log_correct,color = "red",label = "MD")
    ax.plot(time_orbits,msd_log_orbits,"--",color = "blue",label = "GAN")


    ax.set_xlabel("time [ps]",fontsize = 50)
    ax.set_ylabel("MSD [m$^2$]",fontsize = 50)

    ax.legend(fontsize = 45)

    ax.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.axvspan(0,use_step*dt*stepskip*10**-3,color = "coral",alpha = 0.5)

    ax.plot(time_correct,msd_x_log_correct,label = "MD")
    ax.plot(time_orbits,msd_x_log_orbits,"--",label = "GAN")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD x [m$^2$]",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------
    ax.axvspan(0,use_step*dt*stepskip*10**-3,color = "coral",alpha = 0.5)

    ax.plot(time_correct,msd_y_log_correct,color = "red",label = "MD")
    ax.plot(time_orbits,msd_y_log_orbits,"--",color = "blue",label = "GAN")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD y [m$^2$]",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_y.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------
    ax.axvspan(0,use_step*dt*stepskip*10**-3,color = "coral",alpha = 0.5)

    ax.plot(time_correct,msd_z_log_correct,color = "red",label = "MD")
    ax.plot(time_orbits,msd_z_log_orbits,"--",color = "blue",label = "GAN")


    ax.set_xlabel("time [ps]",fontsize = 30)
    ax.set_ylabel("MSD z [m$^2$]",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_z.png")
    plt.close()


    n = np.shape(time)[0]
    print(n)

    time = time_orbits*10**-12

    a = 1/2*(n*np.sum(time*MSD_orbits)-np.sum(time)*np.sum(MSD_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(MSD_orbits)-np.dot(time,MSD_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])




    a = 1/2*(n*np.sum(time*msd_x_log_orbits)-np.sum(time)*np.sum(msd_x_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_x_log_orbits)-np.dot(time,msd_x_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_x [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])



    a = 1/2*(n*np.sum(time*msd_y_log_orbits)-np.sum(time)*np.sum(msd_y_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_y_log_orbits)-np.dot(time,msd_y_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_y [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])


    a = 1/2*(n*np.sum(time*msd_z_log_orbits)-np.sum(time)*np.sum(msd_z_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_z_log_orbits)-np.dot(time,msd_z_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_z [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])




    info.to_csv(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/info.txt",index = False)




    #######################
    #### 潜在変数の描画 ####
    #######################
    latents = np.array(latents)
    print("latent shape",np.shape(latents))

    print("figure shape",np.shape(latents[:,0,3]))
    NUM_MOL = np.shape(latents[:,0,3])[0]

    view_array = [0,1,2,3,4,5,50,100,101,102,103,104,105,150]

    ###
    for l in view_array:
        #figure framework setting

        fig = plt.figure(figsize = (10,10))

        ax = fig.add_subplot(111)

        ax.yaxis.offsetText.set_fontsize(40)
        ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

        #figure detail setting
        for i in range(NUM_MOL):
            ax.scatter(latents[i,l,2],latents[i,l,3])


        ax.set_xlabel("Dimension 3rd",fontsize = 30)
        ax.set_ylabel("Dimension 4th",fontsize = 3)

        ax.minorticks_on()

        ax.tick_params(labelsize = 30, which = "both", direction = "in")
        plt.tight_layout()
        plt.show()

        plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Dimension"+str(l)+"step.png")
        plt.close()
        
        pass
    ####

    """
    ある分子の潜在変数の動きを追ってみる．
    """
    #figure framework setting

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #figure detail setting
    ax.scatter(latents[0,:,2],latents[0,:,3])


    ax.set_xlabel("Dimension 3rd",fontsize = 30)
    ax.set_ylabel("Dimension 4th",fontsize = 30)


    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Dimension No0.png")
    plt.close()
    ##
    #figure framework setting

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #figure detail setting
    ax.scatter(latents[1,:,2],latents[1,:,3])


    ax.set_xlabel("Dimension 3rd",fontsize = 30)
    ax.set_ylabel("Dimension 4th",fontsize = 30)

    print(np.shape(latents))


    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Dimension No1.png")
    plt.close()
    #################################################

    print("distributions")

    #####


    #MSDの計算


    """
    MSDの計算のやり方．

    まず，MDでは50[ps]分のMSDを計算している．
    今，このMD-GANでは，1ステップが10 [fs]に相当する．(step_skip = 15×dt = 2 [fs])

    そのため，同量のMSDを計算する場合，5000ステップ分のデータがあればよい． 1667 × 30 [fs] = 50 [ps]

    次に，MDでは，25000ステップのMSDを計算するにあたって，250ステップずつずらしながら値を抽出していた．
    それらでアンサンブル平均をとる感じ．
    さらに，その250ステップずらす計算をnmsd回繰り返すという事をしてアンサンブル平均をとる．

    つまるところ，100ステップずつ値の抽出開始点をずらして，5000ステップのMSDを取れるだけ取りたいという事．
    そのためにn_pickingを用意している．

    計算手順は次の通り
    1. まず分子の速度（分子のトラジェクトリ）から，取りうる5000ステップのデータをn_picking個抽出する．

    2-a. 1667ステップのデータについて，1ステップ目のデータ（速度値）をピックアップする．

    2-b. 各分子の速度の二乗和の平均を加算していく．即ちここで各ステップのMSDを計算する事になる．

    2-c. n_picking個あるMSDについて，そのアンサンブル平均値をとる．

    2-a,b,cを各ステップずつ繰り返してmsd_log達にくっつけていく．

    3. MSDの三次元の値は次元数で除する．つまり3で割る．

    """
    MOL_NUM = np.shape(orbits)[0]#分子数の算出
    trajectory_length = np.shape(orbits)[1]

    MOL_COL_NUM = np.shape(correct_disp)[0]

    nmsdtime = int(50000/stepskip/dt)
    n_shift_msd = 100
    shift_msd = int(nmsdtime/n_shift_msd)       

    n_picking = 2000                            #アンサンブル数と等価

    length_msd = n_picking*shift_msd+nmsdtime   #使うデータ量

    print(length_msd)

    info_ad = pd.DataFrame(data=[["data lenght for msd",length_msd]],columns = columns2)

    info = pd.concat([info,info_ad])

    ##############################################
    ##################### 1 ######################
    ##############################################
    correct_msd = np.zeros((nmsdtime))
    orbits_msd = np.zeros((nmsdtime))

    for j in range(n_picking):
        #nmsdtime長のトラジェクトリを取れるだけとる．shift_msdはオーバーラップに際してズラすデータ長．
        correct_msd += np.array(msd_log_correct[j*shift_msd:j*shift_msd+nmsdtime]) - msd_log_correct[j*shift_msd]
        orbits_msd += np.array(msd_log_orbits[j*shift_msd:j*shift_msd+nmsdtime]) - msd_log_orbits[j*shift_msd]
        pass

    msd_log_correct = correct_msd/n_picking
    msd_log_orbits = orbits_msd/n_picking
    ####
    correct_msd = np.zeros((nmsdtime))
    orbits_msd = np.zeros((nmsdtime))

    for j in range(n_picking):
        #nmsdtime長のトラジェクトリを取れるだけとる．shift_msdはオーバーラップに際してズラすデータ長．
        correct_msd += np.array(msd_x_log_correct[j*shift_msd:j*shift_msd+nmsdtime] - msd_x_log_correct[j*shift_msd])
        orbits_msd += np.array(msd_x_log_orbits[j*shift_msd:j*shift_msd+nmsdtime] - msd_x_log_orbits[j*shift_msd])
        pass

    msd_x_log_correct = correct_msd/n_picking
    msd_x_log_orbits = orbits_msd/n_picking
    ####
    correct_msd = np.zeros((nmsdtime))
    orbits_msd = np.zeros((nmsdtime))

    for j in range(n_picking):
        #nmsdtime長のトラジェクトリを取れるだけとる．shift_msdはオーバーラップに際してズラすデータ長．
        correct_msd += np.array(msd_y_log_correct[j*shift_msd:j*shift_msd+nmsdtime] - msd_y_log_correct[j*shift_msd])
        orbits_msd += np.array(msd_y_log_orbits[j*shift_msd:j*shift_msd+nmsdtime] - msd_y_log_orbits[j*shift_msd])
        pass

    msd_y_log_correct = correct_msd/n_picking
    msd_y_log_orbits = orbits_msd/n_picking
    ####
    correct_msd = np.zeros((nmsdtime))
    orbits_msd = np.zeros((nmsdtime))

    for j in range(n_picking):
        #nmsdtime長のトラジェクトリを取れるだけとる．shift_msdはオーバーラップに際してズラすデータ長．
        correct_msd += np.array(msd_z_log_correct[j*shift_msd:j*shift_msd+nmsdtime] - msd_z_log_correct[j*shift_msd])
        orbits_msd += np.array(msd_z_log_orbits[j*shift_msd:j*shift_msd+nmsdtime] - msd_z_log_orbits[j*shift_msd])
        pass

    msd_z_log_correct = correct_msd/n_picking
    msd_z_log_orbits = orbits_msd/n_picking
    ####

    # time = np.arange(1,nmsdtime+1)*dt*10**(-3)
    time = np.arange(1,nmsdtime+1)*stepskip*dt*10**(-3)
    print(time)
    print(msd_log_correct)

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_log_correct)

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_correct2.png")
    plt.close()
    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_x_log_correct)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD x [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x_correct.png")
    plt.close()

    n = np.shape(time)[0]
    print(n)

    time = time*10**-12

    a = 1/2*(n*np.sum(time*msd_log_correct)-np.sum(time)*np.sum(msd_log_correct))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_log_correct)-np.dot(time,msd_log_correct)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_md [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])

    time = np.arange(1,nmsdtime+1)*stepskip*dt*10**(-3)
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_log_orbits)

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_predicted2.png")
    plt.close()
    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_x_log_orbits)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD x [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x_predicted2.png")
    plt.close()

    ############################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_y_log_orbits)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD y [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_y_predicted2.png")
    plt.close()

    ############################################
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_z_log_orbits)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD z [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_z_predicted2.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_log_correct,color = "red",label = "MD")
    plt.plot(time,msd_log_orbits,"--",color = "blue",label = "GAN")

    plt.xlim(0,50)

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 50)
    plt.ylabel("MSD [m$^2$]",fontsize = 50)

    plt.legend(fontsize = 50)

    plt.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD2.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_x_log_correct)
    plt.plot(time,msd_x_log_orbits,"--")


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD x [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_x2.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_y_log_correct)
    plt.plot(time,msd_y_log_orbits,"--")


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD y [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_y2.png")
    plt.close()

    ############################################

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,msd_z_log_correct)
    plt.plot(time,msd_z_log_orbits,"--")


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("MSD z [m$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"MSD_z2.png")
    plt.close()


    n = np.shape(time)[0]
    print(n)

    time = time*10**-12

    a = 1/2*(n*np.sum(time*msd_log_orbits)-np.sum(time)*np.sum(msd_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_log_orbits)-np.dot(time,msd_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])




    a = 1/2*(n*np.sum(time*msd_x_log_orbits)-np.sum(time)*np.sum(msd_x_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_x_log_orbits)-np.dot(time,msd_x_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_x [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])



    a = 1/2*(n*np.sum(time*msd_y_log_orbits)-np.sum(time)*np.sum(msd_y_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_y_log_orbits)-np.dot(time,msd_y_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_y [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])


    a = 1/2*(n*np.sum(time*msd_z_log_orbits)-np.sum(time)*np.sum(msd_z_log_orbits))/(n*np.sum(time*time)-np.sum(time)**2)


    b = 1/2*(np.dot(time,time)*np.sum(msd_z_log_orbits)-np.dot(time,msd_z_log_orbits)*np.sum(time))/(n*np.dot(time,time)-np.sum(time)**2)



    print("a : {:.4g}".format(a))
    print("b : {:.4g}".format(b))

    info_ad = pd.DataFrame(data=[["D_pred_z [m$^2$/s]",a]],columns = columns2)

    info = pd.concat([info,info_ad])



    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/MSD.txt",[time,msd_log_correct,msd_log_orbits])

    info.to_csv(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/info2.txt",index = False)


    ########################
    #####  Green-Kubo  #####
    ########################


    print(np.shape(correct_disp))
    print(np.shape(orbits))


    correct_GK_x = np.zeros((nmsdtime))
    correct_GK_y = np.zeros((nmsdtime))
    correct_GK_z = np.zeros((nmsdtime))

    orbits_GK_x = np.zeros((nmsdtime))
    orbits_GK_y = np.zeros((nmsdtime))
    orbits_GK_z = np.zeros((nmsdtime))



    for j in range(n_picking):
        correct_GK_x = correct_GK_x + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,0]*np.broadcast_to(correct_disp[:,j*shift_msd,0][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,0]))),axis = 0)/n_picking
        correct_GK_y = correct_GK_y + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,1]*np.broadcast_to(correct_disp[:,j*shift_msd,1][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,1]))),axis = 0)/n_picking
        correct_GK_z = correct_GK_z + np.average(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,2]*np.broadcast_to(correct_disp[:,j*shift_msd,2][:, np.newaxis],(np.shape(correct_disp[:,j*shift_msd:j*shift_msd+nmsdtime,2]))),axis = 0)/n_picking

        orbits_GK_x = orbits_GK_x + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,0]*np.broadcast_to(orbits[:,j*shift_msd,0][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,0]))),axis = 0)/n_picking
        orbits_GK_y = orbits_GK_y + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,1]*np.broadcast_to(orbits[:,j*shift_msd,1][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,1]))),axis = 0)/n_picking
        orbits_GK_z = orbits_GK_z + np.average(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,2]*np.broadcast_to(orbits[:,j*shift_msd,2][:, np.newaxis],(np.shape(orbits[:,j*shift_msd:j*shift_msd+nmsdtime,2]))),axis = 0)/n_picking
        pass

    ####

    time = np.arange(1,nmsdtime+1)*stepskip*dt*10**(-3)

    correct_GK_x = correct_GK_x*10**10
    correct_GK_y = correct_GK_y*10**10
    correct_GK_z = correct_GK_z*10**10

    correct_GK = (correct_GK_x+correct_GK_y+correct_GK_z)/3

    orbits_GK_x = orbits_GK_x*10**10
    orbits_GK_y = orbits_GK_y*10**10
    orbits_GK_z = orbits_GK_z*10**10

    orbits_GK = (orbits_GK_x+orbits_GK_y+orbits_GK_z)/3

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,correct_GK_x)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_x.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,orbits_GK_x)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_x.png")
    plt.close()



    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,correct_GK_y)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK y [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_y.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,orbits_GK_y)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_y.png")
    plt.close()


    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,correct_GK_z)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK z [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_z.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,orbits_GK_z)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK x [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_z.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,correct_GK,color = "#EF4123")

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"correct_GK.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,orbits_GK)

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"pred_GK.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,orbits_GK,color = "#6495ED")
    plt.plot(time,correct_GK,color = "#EF4123")

    plt.xlim(0,50)

    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("GK [m$^2$/s$^2$]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"pred_and_correct_GK.png")
    plt.close()



    GK_int_correct_x = np.zeros((nmsdtime-1))
    GK_int_correct_y = np.zeros((nmsdtime-1))
    GK_int_correct_z = np.zeros((nmsdtime-1))

    GK_int_orbits_x = np.zeros((nmsdtime-1))
    GK_int_orbits_y = np.zeros((nmsdtime-1))
    GK_int_orbits_z = np.zeros((nmsdtime-1))

    for i in range(0,nmsdtime-1-1):
        GK_int_correct_x[i+1] = GK_int_correct_x[i] + ((correct_GK_x[i]+correct_GK_x[i+1])/2.0)*dt*stepskip*10**(-15)
        GK_int_correct_y[i+1] = GK_int_correct_y[i] + ((correct_GK_y[i]+correct_GK_y[i+1])/2.0)*dt*stepskip*10**(-15)
        GK_int_correct_z[i+1] = GK_int_correct_z[i] + ((correct_GK_z[i]+correct_GK_z[i+1])/2.0)*dt*stepskip*10**(-15)

        GK_int_orbits_x[i+1]  = GK_int_orbits_x[i] + ((orbits_GK_x[i]+orbits_GK_x[i+1])/2.0)*dt*stepskip*10**(-15)
        GK_int_orbits_y[i+1]  = GK_int_orbits_y[i] + ((orbits_GK_y[i]+orbits_GK_y[i+1])/2.0)*dt*stepskip*10**(-15)
        GK_int_orbits_z[i+1]  = GK_int_orbits_z[i] + ((orbits_GK_z[i]+orbits_GK_z[i+1])/2.0)*dt*stepskip*10**(-15)
        pass

    GK_int_correct = (GK_int_correct_x+GK_int_correct_y+GK_int_correct_z)/3
    GK_int_orbits = (GK_int_orbits_x+GK_int_orbits_y+GK_int_orbits_z)/3

    time = np.arange(1,nmsdtime)*stepskip*dt*10**(-3)
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_correct_x)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_int_x.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_orbits_x)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_int_x.png")
    plt.close()

    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_correct_y)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_int_y.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_orbits_y)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_int_y.png")
    plt.close()


    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_correct_z)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_int_z.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    plt.plot(time,GK_int_orbits_z)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_int_z.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)
    plt.plot(time,GK_int_correct,color = "#EF4123")

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_correct_int.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)
    plt.plot(time,GK_int_orbits)

    plt.xlim(0,50)

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_pred_int.png")
    plt.close()

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    #------------------------

    ax.axvspan(int(0.6*nmsdtime)*stepskip*dt*10**(-3),nmsdtime*stepskip*dt*10**(-3),color = "coral",alpha = 0.5)

    plt.plot(time,GK_int_orbits,color = "#6495ED")
    plt.plot(time,GK_int_correct,color = "red")

    plt.xlim(0,50)


    plt.xlabel("time [ps]",fontsize = 30)
    plt.ylabel("D$_{VACF}$ [m$^2$/s]",fontsize = 30)

    # plt.legend(fontsize = 30)

    plt.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/"+"GK_int_pred_and_correct.png")
    plt.close()



    D_PREDICTED = np.average(GK_int_orbits[int(0.6*nmsdtime):])

    info_ad = pd.DataFrame(data=[["D_pred_GK [m$^2$/s]",D_PREDICTED]],columns = columns2)
    info = pd.concat([info,info_ad])


    D_CORRECT = np.average(GK_int_correct[int(0.6*nmsdtime):])

    info_ad = pd.DataFrame(data=[["D_correct_GK [m$^2$/s]",D_CORRECT ]],columns = columns2)
    info = pd.concat([info,info_ad])


    info.to_csv(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/info3.txt",index = False)

    VACF_temp = []
    D_int_temp = []
    for i in time:
        VACF_temp.append(i)
        D_int_temp.append(i)
        pass

    for i in correct_GK:
        VACF_temp.append(i)
        pass

    for i in orbits_GK:
        VACF_temp.append(i)
        pass

    for i in GK_int_correct:
        D_int_temp.append(i)
        pass

    for i in GK_int_orbits:
        D_int_temp.append(i)
        pass


    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/VACF.txt",VACF_temp)
    np.savetxt(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/D_int.txt",D_int_temp)

    ####################
    ### 確率密度分布 ###
    ####################

    correct_disp = correct_disp*10**(5)

    correct_disp_x = correct_disp[:,:,0].reshape(-1)
    correct_disp_y = correct_disp[:,:,1].reshape(-1)
    correct_disp_z = correct_disp[:,:,2].reshape(-1)

    orbits = orbits*10**(5)

    orbits_x = orbits[:,:,0].reshape(-1)
    orbits_y = orbits[:,:,1].reshape(-1)
    orbits_z = orbits[:,:,2].reshape(-1)

    #####
    #figure detail



    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ax.hist(correct_disp_x,bins = 1000,alpha = 0.5,density = True)
    ax.hist(orbits_x,bins = 1000,alpha = 0.5,density = True)

    ax.set_xlabel("velocity [m/s]",fontsize = 30)
    ax.set_ylabel("probability density",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/distribution x.png")
    plt.close()
    #####
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ax.hist(correct_disp_y,bins = 1000,alpha = 0.5,density = True)
    ax.hist(orbits_y,bins = 1000,alpha = 0.5,density = True)

    ax.set_xlabel("velocity [m/s]",fontsize = 30)
    ax.set_ylabel("probability density",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/distribution y.png")
    plt.close()
    #####
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ax.hist(correct_disp_z,bins = 1000,alpha = 0.5,density = True)
    ax.hist(orbits_z,bins = 1000,alpha = 0.5,density = True)

    ax.set_xlabel("velocity [m/s]",fontsize = 30)
    ax.set_ylabel("probability density",fontsize = 30)

    ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 30, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/distribution z.png")
    plt.close()
    #####


    orbits = np.sqrt(np.square(orbits[:,:,0])+np.square(orbits[:,:,1])+np.square(orbits[:,:,2]))
    correct_disp = np.sqrt(np.square(correct_disp[:,:,0])+np.square(correct_disp[:,:,1])+np.square(correct_disp[:,:,2]))

    orbits = orbits.reshape(-1)
    correct_disp = correct_disp.reshape(-1)

    m = M_ar/Na

    vel_sup = 1500 #単位はm/s
    def Maxwell(a,v):
        return (2/np.pi)**(1/2) * (v**2*np.exp(-v**2/(2*a**2))/a**3)


    v = np.linspace(0, vel_sup, 1000)
    a_ = (kb*controlled_T/m)**(1/2)
    p = Maxwell(a_,v)


    #####
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ax.hist(correct_disp,bins = 1000,alpha = 0.5,density = True,label = "MD")
    ax.hist(orbits,bins = 1000,alpha = 0.5,density = True, label = "MD GAN")
    ax.plot(v,p, color = "black")

    ax.set_xlabel("velocity [m/s]",fontsize = 50)
    ax.set_ylabel("probability density",fontsize = 50)

    ax.set_xlim(0,1000)
    ax.set_ylim(0,0.0050)
    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Maxwell1.png")
    plt.close()
    #####
    #figure detail

    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(111)

    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    ax.hist(correct_disp,bins = 1000,alpha = 0.5,density = True,label = "MD")
    ax.hist(orbits,bins = 1000,alpha = 0.5,density = True, label = "MD GAN")

    ax.set_xlabel("velocity [m/s]",fontsize = 50)
    ax.set_ylabel("probability density",fontsize = 50)

    ax.set_xlim(0,1000)
    ax.set_ylim(0,0.0050)

    # ax.legend(fontsize = 30)

    ax.minorticks_on()

    ax.tick_params(labelsize = 45, which = "both", direction = "in")
    plt.tight_layout()
    plt.show()

    plt.savefig(r"/home/s_tanaka/MD-GAN/case8/seed"+str(seed)+"/Maxwell2.png")
    plt.close()
    #####