######################################################################################
#################################### 初期設定 ########################################
######################################################################################

#---   ライブラリの設定（色々弄っていく中で要らなくなったものも混じってる．）
import matplotlib
import numpy as np
import scipy as sp
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import os
import time
import pandas as pd

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 例: 安定したフォントを指定
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 数式のフォントを DejaVu Sans に変更

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
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams["mathtext.fontset"]="stix"

#---   データ読み込み及び必要なパラメータの指定

#フォルダとファイル名指定及びその読み込み
address = r"/home/kawaguchi/data/"               #r"[ファイルが入ってるフォルダー名]"+"/"

DATA_filename = "combined_0.03_3000man.dat" 
data_name = address + DATA_filename

MD_DATA = np.loadtxt(data_name)

parameter_dir = "0.03"
num_dir = ""
result_dir = parameter_dir + "/" + num_dir
model_dir = parameter_dir + "_" + num_dir

#------------------------------------------------------------------------------------


#---   データ読み込み及び必要なパラメ―タ処理2 (主に機械学習でどれだけデータを使うかなどを指定する．)
#データ前処理用の色々
#!!!parameters
data_step = 30000000#MD_DATA.shape[0] #MDのサンプルから取り出してくるデータ長

point_mol_num = 1
dim = 1

###予測データとの比較用にMDデータの処理
correct_disp = np.zeros(shape =(point_mol_num, data_step, dim))

correct_disp[0, :, 0] = np.array(MD_DATA[:data_step, 1])


#図示の際に表示する区間
show_step = 3000

########以下，図示用の処理

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111)

#figure detail

time_step = np.arange(1, np.shape(correct_disp[0])[0] + 1)
time_step_scaled = (time_step - time_step.min()) / (time_step.max() - time_step.min()) * 10  # 0～10にスケール変換

ax.plot(time_step_scaled, correct_disp[0],color = "red")


ax.set_xlabel("Time ns",fontsize = 30)
ax.set_ylabel(r"Heat Flux $\mathrm{W} / \mathrm{m}^2$", fontsize=30)

ax.set_xlim(0, 10)  # x軸を0～10に設定
ax.set_ylim(-1.6e10, 1.6e10)  # y軸の範囲は指定通り

# ax.legend(fontsize = 30)

ax.minorticks_on()
ax.tick_params(labelsize = 27, which = "both", direction = "in")
plt.tight_layout()
# plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/heatflux_true.svg", dpi=600, bbox_inches='tight')
plt.close()  

########################
#####  Green-Kubo  #####
########################

## よく変更する（いずれ上の方に移動するかも）

dt = 1.0 # 時間刻み [fs]     #後々pythonで物理量の評価もしたいなら必要
fs = 1.0E-15
ps = 1.0E-12

timePlot = 10.0 # 相関時間　[ps]
timeSlide = 0.001 # ずらす時間 [ps]   <--------------------------
timeInterval = 0.001 # プロット時間間隔 [ps]

stpRecord = 1 # 


stepPlot = int(timePlot*1.0E+3)
stepSlide = 1
numEnsemble = int(data_step / stepSlide) # <-----------------------もしかして要らない？

print("stepPlot: ", stepPlot)       # 1000 行   熱流束の時刻を 0 にリセットする間隔
print("numEnsemble: ", numEnsemble)     # 100 行    矢印の個数　だと思ってたけど違うかも
print("stepSlide: ", stepSlide)     # 50 行     計算のスタートをずらす間隔

print("correct_disp: ", np.shape(correct_disp))

####

ACF_true = np.zeros((stepPlot))

for i in range(numEnsemble):
    # スライスの範囲がデータサイズを超えないように制御
    start = i * stepSlide
    end = min(start + stepPlot, correct_disp.shape[1])
    n_actual = end - start  # 実際のスライス長

    if n_actual <= 0:
        continue  # 範囲外ならスキップ

    # スライスされたデータ
    disp_slice = correct_disp[:, start:end, 0]

    # ブロードキャスト用のデータを準備 (1D -> 2D)
    ref_data = correct_disp[:, start, 0][:, np.newaxis]
    ref_broadcasted = np.tile(ref_data, (1, n_actual))

    # ACF の計算
    ACF_true[:n_actual] += (disp_slice * ref_broadcasted).sum(axis=0) / numEnsemble

####

time = np.arange(1,stepPlot+1)*dt*fs/ps*stpRecord

# データを結合（2列にする）
ACF_true_data = np.column_stack((time, ACF_true))

# ファイルに保存
np.savetxt(r"/home/kawaguchi/result/" + result_dir + "/ACF_true.dat", ACF_true_data, delimiter=" ", header="time,ACF_true", comments="")


#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

#------------------------

plt.plot(time,ACF_true,color="red")


plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"HFACF $(\mathrm{W} / \mathrm{m}^2)^2$", fontsize=30)

# ax.set_ylim(-3e18, 10e18)

# plt.legend(fontsize = 30)

plt.minorticks_on()

ax.tick_params(labelsize = 27, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ACF_true.svg", dpi=600, bbox_inches='tight')
plt.close()

#--------------------------

T = 100 # [K]
area = 39.2*39.2*10**-20
boltz = 1.3806662*10**(-23)


integration_true = np.zeros((stepPlot-1))
ITR_true = np.zeros((stepPlot-1))

for i in range(0,stepPlot-1-1):

    integration_true[i+1] = integration_true[i] + ((ACF_true[i]+ACF_true[i+1])/2.0)*timeInterval*ps

    pass

for i in range(1, stepPlot-1):

    ITR_true[i] = boltz*T**2/area/integration_true[i]
    
    pass

time = np.arange(1,stepPlot)*dt*fs*stpRecord/ps

# データを結合（2列にする）
ITR_true_data = np.column_stack((time, ITR_true))

# ファイルに保存
np.savetxt(r"/home/kawaguchi/result/" + result_dir + "/ITR_true.dat", ITR_true_data, delimiter=" ", header="time,ITR_true", comments="")


#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# #------------------------
# x, y, z は省略
# #------------------------
# ------------ ITR ---------------

plt.plot(time,ITR_true,color="red")


plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"ITR $\mathrm{K} \cdot \mathrm{m}^2 / \mathrm{W}$", fontsize=30)

# plt.legend(fontsize = 30)

plt.minorticks_on()

ax.tick_params(labelsize = 27, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ITR_true.svg", dpi=600, bbox_inches='tight')
plt.close()

#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# # --------------- ITC -----------------

# ax.axvspan(int(0.6*nmsdtime)*dt*10**(-3)*stpRecord,nmsdtime*dt*10**(-3)*stpRecord,color = "coral",alpha = 0.5)

plt.plot(time,1/ITR_true,color="red")


plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"ITC $\mathrm{W} / (\mathrm{K} \cdot \mathrm{m}^2)$", fontsize=30)

# plt.legend(fontsize = 30)

plt.minorticks_on()

ax.tick_params(labelsize = 27, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ITC_true.svg", dpi=600, bbox_inches='tight')
plt.close()
