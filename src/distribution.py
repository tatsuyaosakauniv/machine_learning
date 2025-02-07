import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ptick
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path  # ファイル存在チェック用
from matplotlib.colors import LinearSegmentedColormap

# --- 図のデフォルト設定の変更 ---
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams["mathtext.fontset"] = "stix"

# i のリスト
i_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]

# 使用するデータファイル名のリスト
data_files = [
    "flow_check_top_1.dat", "flow_check_top_2.dat", "flow_check_top_3.dat",
    "flow_check_bottom_1.dat", "flow_check_bottom_2.dat", "flow_check_bottom_3.dat"
]

# 鮮明な青（#0000FF）から水色（#00FFFF）へのグラデーションを作成
custom_cmap = LinearSegmentedColormap.from_list("custom_blue_cyan", ["#0000FF", "#00FFFF"])

# i_values の数に応じたグラデーションの色を取得
colors_gradient = custom_cmap(np.linspace(1.0, 0.0, len(i_values)))


# グラフを作成
fig, ax = plt.subplots(figsize=(10, 10))

# 軸の設定
ax.yaxis.offsetText.set_fontsize(40)
ax.xaxis.offsetText.set_fontsize(40)  # x軸の指数部分も大きくする
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# データのプロット
for i, color in zip(i_values, colors_gradient):
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

    # 線でつなげてプロット（グラデーションを強調、線を太く）
    ax.plot(bin_centers, hist, color=color, linestyle='-', alpha=0.9, linewidth=2.5, label=f"$\\alpha$ = {i}")

plt.ylim(0, 4.2e-10)

# x=0 の点線を追加
ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

# グラフのラベルを設定
ax.set_xlabel(r"Heat flux $\mathrm{W} / \mathrm{m}^2$", fontsize=25)
ax.set_ylabel('Probability', fontsize=25)

# 軸の設定
ax.tick_params(axis='both', labelsize=27, which="both", direction="in")

# 凡例を追加（背景を白にして視認性を向上）
ax.legend(fontsize=30, loc='upper left', frameon=True)

plt.minorticks_on()
plt.tight_layout()

# グラフを保存
plt.savefig("/home/kawaguchi/result/distribution.svg", dpi=600, bbox_inches='tight')

# 必要に応じて表示
# plt.show()
