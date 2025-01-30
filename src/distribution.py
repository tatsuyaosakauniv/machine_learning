import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ptick
import matplotlib.cm as cm
from pathlib import Path  # ファイル存在チェック用

# --- 図のデフォルト設定の変更 ---
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams["mathtext.fontset"] = "stix"

# i のリスト
i_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]

# 使用するデータファイル名のリスト
data_files = ["flow_check_top_1.dat", "flow_check_top_2.dat", "flow_check_top_3.dat",
            #   "flow_check_top_4.dat", "flow_check_top_5.dat",
              "flow_check_bottom_1.dat", "flow_check_bottom_2.dat","flow_check_bottom_3.dat", 
            #   "flow_check_bottom_4.dat", "flow_check_bottom_5.dat"
              ]

# カラーマップ（赤色のグラデーション）
colors = cm.Reds(np.linspace(0.4, 1.0, len(i_values)))

# グラフを作成
fig, ax = plt.subplots(figsize=(10, 10))

# 軸の設定
ax.yaxis.offsetText.set_fontsize(40)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# データのプロット
for i, color in zip(i_values, colors):
    all_data = []  # iごとのデータを保存するリスト

    for data_file in data_files:
        file_path = Path(f"/home/kawaguchi/data/{i}/{data_file}")

        # ファイルが存在しない場合はスキップ
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        # データ読み込み
        df = pd.read_csv(file_path, sep='\s+', header=None, names=['Column1', 'Column2'])
        all_data.extend(df['Column2'].values)  # データをリストに追加

    if len(all_data) == 0:
        print(f"Warning: No valid data for i={i}, skipping...")
        continue  # データがなければスキップ

    # ヒストグラムのデータを計算 (density=True で確率密度)
    max_abs_value = np.max(np.abs(all_data))
    x_range = (-max_abs_value, max_abs_value)
    hist, bin_edges = np.histogram(all_data, bins=500, range=x_range, density=True)

    # 中心位置を計算
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 線でつなげてプロット（赤のグラデーション）
    ax.plot(bin_centers, hist, color=color, linestyle='-', alpha=0.7, label=f"$\\alpha$ = {i}")

# グラフのラベルを設定
ax.set_xlabel('Heat Flux', fontsize='30')
ax.set_ylabel('Probability Density', fontsize='30')

# 凡例を追加
ax.legend(fontsize=30, loc='upper left')  # 右上に凡例を配置

plt.minorticks_on()

ax.tick_params(labelsize = 30, which = "both", direction = "in")
plt.tight_layout()

# 軸の設定
ax.yaxis.offsetText.set_fontsize(40)
ax.xaxis.offsetText.set_fontsize(40)  # x軸の指数部分も大きくする

ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# x軸とy軸のラベルのフォントサイズを設定
ax.tick_params(axis='both', labelsize=30)

# グラフを保存
plt.savefig("/home/kawaguchi/result/distribution.svg", dpi=600, bbox_inches='tight')

# 必要に応じて表示
# plt.show()
