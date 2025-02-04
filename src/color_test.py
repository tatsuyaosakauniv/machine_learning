import matplotlib.pyplot as plt
import numpy as np

# x の範囲（第一象限と第四象限）
x = np.linspace(0, 10, 100)  # x >= 0 のみ

# y の計算
y1 = x / 2   # 第一象限
y2 = -x / 2  # 第四象限

# プロット
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y1, label=r"$y = \frac{x}{2}$", color="blue", linewidth=2)
ax.plot(x, y2, label=r"$y = -\frac{x}{2}$", color="red", linewidth=2)

# 軸の設定
ax.axhline(0, color='black', linewidth=1)  # x軸
ax.axvline(0, color='black', linewidth=1)  # y軸

# ラベルと凡例
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
# 凡例を追加（背景を白にして視認性を向上）
ax.legend(fontsize=30, loc='upper left')

# 範囲指定（第一象限と第四象限）
ax.set_xlim(0, 10)
ax.set_ylim(-5, 5)

plt.grid()
plt.show()
