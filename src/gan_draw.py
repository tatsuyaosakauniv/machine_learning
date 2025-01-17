import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import numpy as np
from gan_config import *

def plot_results(orbits, correct_disp, AVERAGE_DATA, STD_DATA):
    orbits[:, :, 0] = orbits[:, :, 0] * STD_DATA + AVERAGE_DATA
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    time_step = np.arange(1, np.shape(orbits[0])[0] + 1)
    time_step_scaled = (time_step - time_step.min()) / (time_step.max() - time_step.min()) * 10
    ax.plot(time_step_scaled, orbits[0], color="blue")
    ax.set_xlabel("Time ns", fontsize=30)
    ax.set_ylabel("Heat Flux W/m$^2$", fontsize=30)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6e10, 1.6e10)
    ax.minorticks_on()
    ax.tick_params(labelsize=30, which="both", direction="in")
    plt.tight_layout()
    plt.savefig(r"/home/kawaguchi/result/heatflux_pred.png")
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.offsetText.set_fontsize(40)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.plot(time_step_scaled, correct_disp[0], color="red")
    ax.set_xlabel("Time ns", fontsize=30)
    ax.set_ylabel("Heat Flux W/m$^2$", fontsize=30)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6e10, 1.6e10)
    ax.minorticks_on()
    ax.tick_params(labelsize=30, which="both", direction="in")
    plt.tight_layout()
    plt.savefig(r"/home/kawaguchi/result/heatflux_true.png")
    plt.close()

# 結果のプロット
plot_results(orbits, correct_disp, AVERAGE_DATA, STD_DATA)