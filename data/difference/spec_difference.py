import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import os

# ===============================
# 波长范围
# ===============================
LAMBDA_MIN = 400
LAMBDA_MAX = 700

def crop(df):
    return df[(df["wavelength_nm"] >= LAMBDA_MIN) &
              (df["wavelength_nm"] <= LAMBDA_MAX)]

# ===============================
# 路径
# ===============================
script_folder = os.path.dirname(os.path.abspath(__file__))

# ===============================
# 光源
# ===============================
source_path = os.path.join(script_folder, "source.csv")

df_source = pd.read_csv(source_path)
df_source.columns = df_source.columns.str.lower()
df_source = df_source.rename(columns={
    "wavelength": "wavelength_nm",
    "intensity": "intensity"
})

df_source = crop(df_source)
df_source = df_source.sort_values("wavelength_nm").groupby(
    "wavelength_nm", as_index=False).mean()

wl = df_source["wavelength_nm"].values * 1e-9
I0 = df_source["intensity"].values

spectrum_func = interp1d(wl, I0, kind="cubic", fill_value="extrapolate")

lam = np.linspace(400e-9, 700e-9, 20000)

# ===============================
# 模型（θ3/n3已删除）
# ===============================
def alpha_lambda(lam, con, length):
    K = 2.1648e7 * (1e-9)**2 * 10 * 1000 * 0.001
    lam0 = 146e-9
    return K * length * con / (lam**2 - lam0**2)

def transmission(con, length, theta1, theta2, n1, n2):
    alpha = alpha_lambda(lam, con, length)
    Iin = spectrum_func(lam)

    f1 = np.cos(np.deg2rad(n1 * alpha - theta1))**2
    f2 = np.cos(np.deg2rad(n2 * alpha - (theta2 - theta1)))**2

    return Iin * f1 * f2

# ===============================
# ⭐ 修复版 FWHM（核心）
# ===============================
def fwhm(x, y, peak_index):

    peak = y[peak_index]
    half = peak / 2

    # 左侧 crossing
    l = peak_index
    while l > 0 and y[l] > half:
        l -= 1
    if l == 0:
        return None

    # 右侧 crossing
    r = peak_index
    while r < len(y)-1 and y[r] > half:
        r += 1
    if r == len(y)-1:
        return None

    # ===== 插值（关键）=====
    x1, x2 = x[l], x[l+1]
    y1, y2 = y[l], y[l+1]
    xl = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

    x1, x2 = x[r-1], x[r]
    y1, y2 = y[r-1], y[r]
    xr = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

    return xl, xr - xl

# ===============================
# SMSR
# ===============================
def smsr(I):
    p, _ = find_peaks(I, prominence=0.02 * np.max(I))
    if len(p) < 2:
        return None
    h = np.sort(I[p])
    return 10 * np.log10(h[-1] / h[-2])

# ===============================
# 单文件夹处理
# ===============================
def process(folder):

    path = os.path.join(script_folder, folder)
    target = float(folder)

    csv_file = os.path.join(path, f"{int(target)}.csv")
    param_file = os.path.join(path, "params.csv")

    # ===============================
    # params
    # ===============================
    par = pd.read_csv(param_file)
    theta1, theta2, n1, n2 = par.iloc[0].values

    # ===============================
    # exp
    # ===============================
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "wavelength": "wavelength_nm",
        "intensity": "intensity"
    })

    df = crop(df)
    df = df.sort_values("wavelength_nm")

    lam_exp = df["wavelength_nm"].values * 1e-9
    I_exp = savgol_filter(df["intensity"].values, 11, 3)

    # ===============================
    # theory
    # ===============================
    I_th = transmission(0.7, 0.2, theta1, theta2, n1, n2)
    I_th = savgol_filter(I_th, 101, 3)

    # ===============================
    # peak（修复：真正主峰）
    # ===============================
    p_th, _ = find_peaks(I_th, prominence=0.02 * np.max(I_th))
    p_ex, _ = find_peaks(I_exp, prominence=0.02 * np.max(I_exp))

    if len(p_th) == 0 or len(p_ex) == 0:
        print(f"[WARN] No peaks in {folder}")
        return None

    main_th = p_th[np.argmax(I_th[p_th])]
    main_ex = p_ex[np.argmax(I_exp[p_ex])]

    # ===============================
    # center
    # ===============================
    center_th = lam[np.argmax(I_th)]
    center_ex = lam_exp[np.argmax(I_exp)]

    # ===============================
    # FWHM（稳定版）
    # ===============================
    f_th = fwhm(lam * 1e9, I_th, main_th)
    f_ex = fwhm(lam_exp * 1e9, I_exp, main_ex)

    if f_th is None:
        fw_th = np.nan
    else:
        _, fw_th = f_th

    if f_ex is None:
        fw_ex = np.nan
    else:
        _, fw_ex = f_ex

    # ===============================
    # SMSR
    # ===============================
    s_th = smsr(I_th)
    s_ex = smsr(I_exp)

    # ===============================
    # RMSE
    # ===============================
    interp = interp1d(lam * 1e9, I_th, fill_value="extrapolate")
    I_th_exp = interp(lam_exp * 1e9)

    rmse = np.sqrt(np.mean(
        (I_exp / np.max(I_exp) - I_th_exp / np.max(I_th_exp)) ** 2
    ))

    # ===============================
    # result
    # ===============================
    result = {
        "target": target,
        "center_err": abs(center_ex - center_th),
        "fwhm_err": abs(fw_ex - fw_th) if not np.isnan(fw_ex) else np.nan,
        "smsr_err": None if (s_th is None or s_ex is None)
        else abs(s_ex - s_th),
        "rmse": rmse,
        "center_exp": center_ex,
        "center_th": center_th,
        "fwhm_exp": fw_ex,
        "fwhm_th": fw_th,
        "smsr_exp": s_ex,
        "smsr_th": s_th
    }

    # ===============================
    # PDF（红蓝双y轴）
    # ===============================
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(lam * 1e9, I_th, color="blue", linewidth=2, label="Theory")
    ax2.plot(lam_exp * 1e9, I_exp, color="red", linewidth=2, label="Experiment")

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Theory Intensity", color="blue")
    ax2.set_ylabel("Experiment Intensity", color="red")

    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"{folder} Theory vs Experiment")
    plt.tight_layout()

    plt.savefig(os.path.join(path, "target_difference.pdf"), dpi=300)
    plt.close()

    # ===============================
    # CSV
    # ===============================
    pd.DataFrame([result]).to_csv(
        os.path.join(path, "target_difference.csv"),
        index=False
    )

    return result

# ===============================
# 批处理
# ===============================
results = []

for f in os.listdir(script_folder):
    if f.isdigit():
        print("Processing:", f)
        r = process(f)
        if r:
            results.append(r)

pd.DataFrame(results).to_csv(
    os.path.join(script_folder, "all_targets_difference.csv"),
    index=False
)

print("DONE")