import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを読み込みます（スペース区切りのデータとして読み込む）
df = pd.read_csv(r"/home/kawaguchi/data/flow_check_top_1211.dat", sep='\s+', header=None, names=['Column1', 'Column2'])

# データフレームの内容を確認します
print(df.head())

# データフレームの列数を確認します
num_columns = df.shape[1]
print(f"Number of columns in the dataframe: {num_columns}")

# 2列目のデータを抽出します（列数が2以上の場合）
if num_columns >= 2:
    column2 = df['Column2']
    
    # x軸の中心を0に設定し、範囲を絶対値の最大値から自動的に決定します
    max_abs_value = np.max(np.abs(column2))
    x_range = (-max_abs_value, max_abs_value)
    
    # ヒストグラムのデータを計算します
    hist, bin_edges = np.histogram(column2, bins=30, range=x_range)
    
    # 中心位置を計算します
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 点のヒストグラムをプロットします
    plt.scatter(bin_centers, hist, marker='o')
    
    # グラフのラベルを設定します
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of 2nd Column Data')
    
    # グラフを保存します
    plt.savefig(r"/home/kawaguchi/result/distribution.png")
    
    # グラフを表示します（必要に応じて）
    # plt.show()
else:
    print("Error: The dataframe does not contain at least 2 columns.")