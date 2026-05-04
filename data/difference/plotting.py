import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# =========================
# 论文级绘图风格
# =========================
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.2,
    "mathtext.fontset": "stix",
    "font.family": "Times New Roman"
})

# =========================
# 1. 读取 + 截断
# =========================
def load_spectrum(file_path):

    df = pd.read_csv(file_path)

    wl = df.iloc[:, 0].values
    I = df.iloc[:, 1].values

    # 清理 NaN / inf
    mask = np.isfinite(wl) & np.isfinite(I)
    wl, I = wl[mask], I[mask]

    # 400–700 nm 截断
    mask = (wl >= 400) & (wl <= 700)
    wl, I = wl[mask], I[mask]

    # 排序
    idx = np.argsort(wl)
    wl, I = wl[idx], I[idx]

    return wl, I


# =========================
# 2. 插值 + 平滑
# =========================
def smooth_spectrum(wl, I, num=2000):

    wl_min, wl_max = wl.min(), wl.max()

    f = interp1d(
        wl,
        I,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate"
    )

    wl_new = np.linspace(wl_min, wl_max, num)
    I_new = f(wl_new)

    window = min(51, len(I_new)//2*2-1)
    if window < 5:
        window = 5

    I_smooth = savgol_filter(I_new, window_length=window, polyorder=3)

    return wl_new, I_smooth


# =========================
# 3. 绘图（真实光强版）
# =========================
def plot_spectrum(file_path,
                  save_name="spectrum_400_700nm.pdf"):

    wl, I = load_spectrum(file_path)
    wl_s, I_s = smooth_spectrum(wl, I)

    plt.figure(figsize=(7.5, 4.5))

    # 平滑曲线（真实光强）
    plt.plot(
        wl_s,
        I_s,
        color="#1f77b4",
        linewidth=2.0,
        label="Smoothed Spectrum"
    )

    # 原始数据（真实光强）
    plt.scatter(
        wl,
        I,
        s=10,
        color="gray",
        alpha=0.4,
        label="Raw data"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")  # ⭐关键：真实光强

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(save_name, bbox_inches="tight")
    plt.show()


# =========================
# 运行
# =========================
if __name__ == "__main__":
    plot_spectrum(
        file_path="source.csv",
        save_name="source_spectrum_400_700nm.pdf"
    )