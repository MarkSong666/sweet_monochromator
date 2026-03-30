import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
import pandas as pd
import os

# ===============================
# 读取光谱
# ===============================
script_folder = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_folder, '../data/spec_source.csv')

df = pd.read_csv(csv_file_path)
df = df.sort_values(by="wavelength_nm")
df = df.groupby("wavelength_nm", as_index=False).mean()

wl_data = df["wavelength_nm"].values * 1e-9
I_data = df["intensity"].values

spectrum_func = interp1d(
    wl_data,
    I_data,
    kind='cubic',
    fill_value="extrapolate"
)

lam = np.linspace(400e-9, 750e-9, 20000)

# ===============================
# 旋光模型
# ===============================
def alpha_lambda(lam, con, length):
    K = 2.1648e7 * (1e-9)**2 * 10 * 1000 * 0.001
    lam0 = 146e-9
    return K * length * con / (lam**2 - lam0**2)

def transmission(con, length, theta1, theta2, theta3, n1, n2, n3):
    alpha = alpha_lambda(lam, con, length)
    Iin = spectrum_func(lam)

    f1 = np.cos(np.deg2rad(n1*alpha - theta1))**2
    f2 = np.cos(np.deg2rad(n2*alpha - (theta2 - theta1)))**2
    f3 = np.cos(np.deg2rad(n3*alpha - (theta3 - theta2)))**2

    return Iin * f1 * f2 * f3

# ===============================
# FWHM
# ===============================
def calculate_precise_fwhm(x, y, peak_index):
    peak_height = y[peak_index]
    half_max = peak_height / 2

    left = peak_index
    while left > 0 and y[left] > half_max:
        left -= 1
    if left == 0:
        return None

    x1, x2 = x[left], x[left+1]
    y1, y2 = y[left], y[left+1]
    left_interp = x1 + (half_max - y1)*(x2-x1)/(y2-y1)

    right = peak_index
    while right < len(y)-1 and y[right] > half_max:
        right += 1
    if right == len(y)-1:
        return None

    x1, x2 = x[right-1], x[right]
    y1, y2 = y[right-1], y[right]
    right_interp = x1 + (half_max - y1)*(x2-x1)/(y2-y1)

    return x[peak_index], right_interp-left_interp, half_max, left_interp, right_interp

# ===============================
# SMSR
# ===============================
def calculate_smsr(Iout):
    peaks, _ = find_peaks(Iout, prominence=0.02*np.max(Iout))
    if len(peaks) < 2:
        return None
    heights = Iout[peaks]
    sorted_h = np.sort(heights)
    return 10*np.log10(sorted_h[-1]/sorted_h[-2])

# ===============================
# 初始化图像
# ===============================
fig, ax = plt.subplots(figsize=(16,8))
plt.subplots_adjust(bottom=0.45)

Iout = transmission(0.7,0.2,0,0,0,1,1,1)
main_line, = ax.plot(lam*1e9, Iout, lw=2, color='black', label="Current")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity")
ax.legend()

peak_markers = []
fwhm_lines = []
saved_lines = []
colors = plt.cm.tab10.colors
color_index = 0

# ===============================
# 创建滑块
# ===============================
def create_slider(label, vmin, vmax, vinit, y):
    ax_s = plt.axes([0.2, y, 0.2, 0.015])
    slider = Slider(ax_s, label, vmin, vmax, valinit=vinit)

    ax_box = plt.axes([0.43, y, 0.06, 0.02])
    textbox = TextBox(ax_box, '', initial=f"{vinit:.2f}")

    def update_text(val):
        textbox.set_val(f"{val:.2f}")
    slider.on_changed(update_text)

    def submit(text):
        try:
            slider.set_val(np.clip(float(text), vmin, vmax))
        except:
            pass
    textbox.on_submit(submit)

    return slider

s_con = create_slider("Concentration",0,2,0.7,0.32)
s_t1  = create_slider("Theta1",0,360,0,0.29)
s_t2  = create_slider("Theta2",0,360,0,0.26)
s_t3  = create_slider("Theta3",0,360,0,0.23)
s_n1  = create_slider("n1",0,10,1,0.20)
s_n2  = create_slider("n2",0,10,1,0.17)
s_n3  = create_slider("n3",0,10,1,0.14)

# ===============================
# 更新函数
# ===============================
def update(val):

    global peak_markers, fwhm_lines

    Iout = transmission(
        s_con.val,0.2,
        s_t1.val,s_t2.val,s_t3.val,
        s_n1.val,s_n2.val,s_n3.val
    )

    main_line.set_ydata(Iout)
    ax.set_ylim(np.min(Iout)*0.95, np.max(Iout)*1.05)

    for m in peak_markers:
        m.remove()
    for l in fwhm_lines:
        l.remove()
    peak_markers.clear()
    fwhm_lines.clear()

    peaks,_ = find_peaks(Iout,prominence=0.05*np.max(Iout))
    title_text = []

    for p in peaks:
        res = calculate_precise_fwhm(lam*1e9,Iout,p)
        if res is None:
            continue
        center,fwhm,half,l_int,r_int = res

        marker = ax.plot(center,Iout[p],'ro')
        peak_markers.extend(marker)

        line = ax.hlines(half,l_int,r_int,colors='red',linestyles='--')
        fwhm_lines.append(line)

        title_text.append(f"{center:.1f}nm FWHM={fwhm:.2f}")

    smsr = calculate_smsr(Iout)
    if smsr is not None:
        title_text.append(f"SMSR={smsr:.2f} dB")

    ax.set_title(" | ".join(title_text))
    fig.canvas.draw_idle()

for s in [s_con,s_t1,s_t2,s_t3,s_n1,s_n2,s_n3]:
    s.on_changed(update)

# ===============================
# 多曲线按钮
# ===============================
ax_add = plt.axes([0.6,0.1,0.1,0.03])
btn_add = Button(ax_add,"Add Curve")

ax_clear = plt.axes([0.72,0.1,0.1,0.03])
btn_clear = Button(ax_clear,"Clear All")

def add_curve(event):
    global color_index
    Iout = transmission(
        s_con.val,0.2,
        s_t1.val,s_t2.val,s_t3.val,
        s_n1.val,s_n2.val,s_n3.val
    )
    color = colors[color_index % len(colors)]
    color_index += 1
    line, = ax.plot(lam*1e9,Iout,color=color,alpha=0.8)
    saved_lines.append(line)
    fig.canvas.draw_idle()

def clear_all(event):
    global saved_lines,color_index
    for l in saved_lines:
        l.remove()
    saved_lines=[]
    color_index=0
    fig.canvas.draw_idle()

btn_add.on_clicked(add_curve)
btn_clear.on_clicked(clear_all)

# ===============================
# 进度条
# ===============================
ax_progress = plt.axes([0.2,0.05,0.4,0.02])
progress_bar = ax_progress.barh([0],[0])
ax_progress.set_xlim(0,100)
ax_progress.set_xticks([])
ax_progress.set_yticks([])
ax_progress.set_title("Optimization Progress")

# ===============================
# 目标优化
# ===============================
ax_target = plt.axes([0.65,0.05,0.1,0.03])
target_box = TextBox(ax_target,"Target nm",initial="550")

ax_opt = plt.axes([0.78,0.05,0.1,0.03])
btn_opt = Button(ax_opt,"Optimize")

import threading

def optimize_target(event):

    def run_optimization():

        target_nm = float(target_box.text)
        target = target_nm*1e-9
        maxiter = 200

        progress_bar[0].set_width(0)

        def objective(params):
            theta1,theta2,theta3,n2,n3 = params
            Iout = transmission(
                s_con.val,0.2,
                theta1,theta2,theta3,
                s_n1.val,n2,n3
            )
            peaks,_ = find_peaks(Iout,prominence=0.02*np.max(Iout))
            if len(peaks)<2:
                return 1e6
            peak_lams = lam[peaks]
            peak_heights = Iout[peaks]
            idx = np.argmin(np.abs(peak_lams-target))
            if peak_heights[idx] < np.max(peak_heights):
                return 1e5
            sorted_h = np.sort(peak_heights)
            smsr = 10*np.log10(sorted_h[-1]/sorted_h[-2])
            shift = abs(peak_lams[idx]-target)*1e9

            main_peak_idx = peaks[idx]

            res = calculate_precise_fwhm(lam*1e9, Iout, main_peak_idx)
            if res == None:
                return 1e6
            _, fwhm, _, _, _ = res

            return -50*smsr + 50*shift + 5*fwhm

        iteration = {"count":0}

        def callback(xk, convergence):
            iteration["count"]+=1
            progress = iteration["count"]/maxiter*100
            progress_bar[0].set_width(progress)
            fig.canvas.draw_idle()

        bounds=[(0,360),(0,360),(0,360),(0,10),(0,10)]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            callback=callback,
            popsize=50,
            mutation=(0.5,1.7),
            recombination=1.4,
            workers=1,
            strategy='best1bin'
        )

        # 更新slider必须回到主线程
        def update_sliders():
            s_t1.set_val(result.x[0])
            s_t2.set_val(result.x[1])
            s_t3.set_val(result.x[2])
            s_n2.set_val(result.x[3])
            s_n3.set_val(result.x[4])
            progress_bar[0].set_width(100)
            fig.canvas.draw_idle()

        fig.canvas.manager.window.after(0, update_sliders)

    threading.Thread(target=run_optimization, daemon=True).start()

btn_opt.on_clicked(optimize_target)

plt.show()