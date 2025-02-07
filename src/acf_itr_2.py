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

parameter_dir = "final_2.0"
num_dir = "0.1"
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

data = np.loadtxt(r"/home/kawaguchi/result/" + result_dir + "/ACF_true.dat", skiprows=1)
# 列ごとに分割
md_time = data[:, 0]  # 1列目
ACF_true = data[:, 1]  # 2列目
data = np.loadtxt(r"/home/kawaguchi/result/" + result_dir + "/ACF_pred.dat", skiprows=1)
# 列ごとに分割
ACF_pred = data[:, 1]  # 2列目

########以下，図示用の処理

#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

#------------------------

plt.plot(md_time,ACF_true,color="red", label="MD")
plt.plot(md_time,ACF_pred,color="blue", label="GANs")

plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"ACF $(\mathrm{W} / \mathrm{m}^2)^2$", fontsize=30)

# 凡例を追加
ax.legend(fontsize=30, loc='upper right', frameon=True)

plt.minorticks_on()

ax.tick_params(labelsize = 30, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ACF_pred_and_true.svg", dpi=600, bbox_inches='tight')
plt.close()

#--
data = np.loadtxt(r"/home/kawaguchi/result/" + result_dir + "/ITR_true.dat", skiprows=1)
# 列ごとに分割
md_time = data[:, 0]  # 1列目
ITR_true = data[:, 1]  # 2列目
data = np.loadtxt(r"/home/kawaguchi/result/" + result_dir + "/ITR_pred.dat", skiprows=1)
# 列ごとに分割
ITR_pred = data[:, 1]  # 2列目

#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# ax.axvspan(int(0.6*stepPlot)*dt*10**(-3)*stpRecord,stepPlot*dt*10**(-3)*stpRecord,color = "coral",alpha = 0.5)

plt.plot(md_time,ITR_true,color="red", label="MD")
plt.plot(md_time,ITR_pred,color="blue", label="GANs")

plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"ITR $\mathrm{K} \cdot \mathrm{m}^2 / \mathrm{W}$", fontsize=30)

# 凡例を追加
ax.legend(fontsize=30, loc='upper right', frameon=True)

plt.minorticks_on()

ax.tick_params(labelsize = 30, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ITR_pred_and_true.svg", dpi=600, bbox_inches='tight')
plt.close()

#figure detail

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)

ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# ax.axvspan(int(0.6*nmsdtime)*dt*10**(-3)*stpRecord,nmsdtime*dt*10**(-3)*stpRecord,color = "coral",alpha = 0.5)

plt.plot(md_time,1/ITR_true,color="red", label="MD")
plt.plot(md_time,1/ITR_pred,color="blue", label="GANs")

plt.xlabel("Time ps",fontsize = 30)
plt.ylabel(r"ITC $\mathrm{W} / (\mathrm{K} \cdot \mathrm{m}^2)$", fontsize=30)

# 凡例を追加
ax.legend(fontsize=30, loc='upper right', frameon=True)

plt.minorticks_on()

ax.tick_params(labelsize = 30, which = "both", direction = "in")
plt.tight_layout()
plt.show()

plt.savefig(r"/home/kawaguchi/result/" + result_dir + "/ITC_pred_and_true.svg", dpi=600, bbox_inches='tight')
plt.close()