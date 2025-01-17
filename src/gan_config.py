import os

#---   ライブラリの設定
import matplotlib
import numpy as np
import random
import pandas as pd
import tensorflow as tf

#---   共通設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#---   シード値の設定
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    return

set_seed(1)

#---   データ読み込み及び必要なパラメータの指定
address = r"/home/kawaguchi/data/"
DATA_filename = "flow_check_top_0108.dat"
data_name = address + DATA_filename

data_step = 3000000
use_step = 1000000

#!!!!!!!!!!!!!!!!!!テキストファイル用!!!!!!!!!!!!!!!!!!!!!!!
columns2 = ["parameter","value"]
info = pd.DataFrame(data = [["data_step",data_step]],columns= columns2)
info_ad = pd.DataFrame(data=[["use_step",use_step]],columns = columns2)
info = pd.concat([info,info_ad])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

point_mol_num = 1
dim = 1

sequence_length = 500
batch_size = int(use_step / sequence_length)
data_length = int(use_step / sequence_length)
iteration_all = 20000

random_uniform_inf = 0
random_uniform_sup = 1.0
means2 = 0
stds2 = 1.0
hidden_node = 128
discriminator_extra_steps = 5
gen_lr = 0.00005
disc_lr = gen_lr / discriminator_extra_steps