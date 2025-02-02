import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ptick
from matplotlib.ticker import MultipleLocator

# --- フォントとスタイルの設定 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 30

# --- データの読み込み ---
data = np.loadtxt("press_itr_compare_nemd.dat")

# 列データの取得
labels = data[:, 0]  # 1列目（ラベル）
x = data[:, 1]       # 2列目（圧力）
y_values = data[:, 2:6]  # 3,4,5,6列目（ITR）

# --- カラーマップの設定（赤→山吹色のグラデーション）---
# 赤と山吹色を指定したカスタムカラーマップを作成
colors = mcolors.LinearSegmentedColormap.from_list("red_yellow", ["#FF0000", "#FFEA00"])

num_rows = len(data)  # データの行数を取得
colors_gradient = colors(np.linspace(0.2, 1.0, num_rows))  # グラデーションを適用

# --- グラフの作成 ---
fig, ax = plt.subplots(figsize=(10, 10))

# データ行ごとに散布図をプロット
for i in range(num_rows):
    label_text = f"$\\alpha = {labels[i]:.2f}$"  # ラベルを α = 値 の形式に
    plt.scatter([x[i]] * 4, y_values[i, :], label=label_text, color=colors_gradient[i], alpha=0.9, s=80)

# 軸ラベルと設定
ax.set_xlabel("Pressure $\mathrm{MPa}$", fontsize=30)
ax.set_ylabel("ITR $\mathrm{m}^2 \\cdot \mathrm{K} \\, / \\,\mathrm{W}$", fontsize=30)

# --- X軸の範囲を 0 〜 50 に設定 ---
ax.set_xlim(16, 50)  # ここで範囲を指定

# 軸のスタイル調整
ax.tick_params(labelsize=30, which="both", direction="in")
plt.minorticks_on()

# --- X軸の目盛りを10刻みにする ---
ax.xaxis.set_major_locator(MultipleLocator(10))  # 10刻みの目盛り
ax.xaxis.set_minor_locator(MultipleLocator(2))  # 補助目盛りを2刻みに設定

# 凡例の設定
ax.legend(fontsize=30, loc='upper left')  # 左上に配置

# 指数表記を適用（軸のフォーマット）
ax.yaxis.offsetText.set_fontsize(30)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# グラフの保存
plt.savefig("ITR_vs_Pressure.svg", dpi=600, bbox_inches='tight')

# 必要に応じて表示
# plt.show()
