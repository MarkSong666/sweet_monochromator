import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ====================== 参数配置 ======================
script_folder = os.path.dirname(os.path.abspath(__file__))
# 路径适配你的目录结构
LIGHT_FILE = os.path.join(script_folder, '../data/光源光谱粗测.xlsx')
DATA_FILE = os.path.join(script_folder, '../data/各浓度光谱.xlsx')
L = 2                     # 光程长度 (dm)
K_THEORY = 2.1648e7         # 理论 K 值 (deg·nm²·dm⁻¹·g⁻¹·ml)
LAMBDA0_THEORY = 146        # 理论 λ₀ (nm)
SMOOTH_ALPHA = True
SMOOTH_RATIO = True
# 枚举参数
WAVE_START = 350            # 起始波长 (nm)
WAVE_END = 750              # 结束波长 (nm)
WINDOW_WIDTH = 75           # 窗口宽度 (nm)
STEP = 10                   # 步长 (nm)
# ========== 筛选参数（可调整） ==========
MAX_K_REL_ERR = 0.2         # K 与理论值的最大相对误差（20%）
MAX_LAMBDA0_ABS_ERR = 50    # λ₀ 与理论值的最大绝对误差 (nm)
MAX_K_REL_ERR_UNC = 0.5     # K 的误差与 K 的比值上限（50%）
MAX_LAMBDA0_REL_ERR_UNC = 0.5  # λ₀ 的误差与 λ₀ 的比值上限（50%）
MIN_R2 = 0.8                # R² 下限

# ====================== 辅助函数 ======================
def load_spectrum_excel(filepath):
    """加载光谱Excel，返回波长和强度数组"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}，请检查路径和文件名")
    # 强制指定openpyxl引擎，避免依赖错误
    df = pd.read_excel(filepath, header=None, engine='openpyxl')
    wave = df.iloc[:, 0].dropna().values
    intensity = df.iloc[:, 1].dropna().values
    if len(wave) != len(intensity):
        raise ValueError(f"文件{filepath}的波长和强度数据长度不一致")
    return wave, intensity

def theoretical_alpha(wavelengths, c, l):
    """计算理论旋光角α"""
    lam = np.asarray(wavelengths)
    return l * c * K_THEORY / (lam**2 - LAMBDA0_THEORY**2)

def compute_alpha_from_sample(light_file, sample_wave, sample_int, theta0, c, l):
    """从样品光谱计算旋光角α"""
    # 加载光源光谱并插值
    wave_light, I0 = load_spectrum_excel(light_file)
    f_interp = interp1d(wave_light, I0, kind='linear', fill_value='extrapolate')
    I0_interp = f_interp(sample_wave)
    
    # 避免除零错误
    I0_interp[I0_interp == 0] = np.finfo(float).eps
    R = sample_int / I0_interp
    
    # 平滑强度比
    if SMOOTH_RATIO and len(R) >= 5:
        window = min(11, len(R) if len(R)%2==1 else len(R)-1)
        if window >= 3:
            R = savgol_filter(R, window, 3)
    
    # 归一化并限制范围，避免arccos报错
    R_norm = R / np.nanmax(R)
    R_norm = np.clip(R_norm, 1e-8, 1-1e-8)  # 严格限制在(0,1)内
    
    # 计算旋光角
    phi = np.arccos(np.sqrt(R_norm))
    phi_deg = np.rad2deg(phi)
    alpha_th = theoretical_alpha(sample_wave, c, l)
    alpha = np.zeros_like(sample_wave)
    
    # 周期折叠匹配理论值
    for i, (lam_i, phi_i, th_i) in enumerate(zip(sample_wave, phi_deg, alpha_th)):
        candidates = []
        for sign in (-1, 1):
            for k in range(-3, 4):
                delta = sign * phi_i + k * 180.0
                alpha_candidate = theta0 + delta
                if 0 <= alpha_candidate <= 720:
                    candidates.append(alpha_candidate)
        candidates = list(set(candidates))
        if not candidates:
            alpha[i] = np.nan
            continue
        best = min(candidates, key=lambda x: abs(x - th_i))
        alpha[i] = best
    
    # 平滑最终旋光角
    if SMOOTH_ALPHA and len(alpha) >= 5:
        window = min(11, len(alpha) if len(alpha)%2==1 else len(alpha)-1)
        if window >= 3:
            alpha = savgol_filter(alpha, window, 3)
    
    # 剔除NaN值
    valid_mask = ~np.isnan(alpha)
    return sample_wave[valid_mask], alpha[valid_mask]

def fit_alpha_linear(lam, alpha, l, c):
    """
    滑动窗口线性拟合，返回最优拟合结果
    返回 (K, K_err, lambda0, lambda0_err, R2, best_start, best_end)
    """
    best_K = None
    best_K_err = None
    best_lam0 = None
    best_lam0_err = None
    best_r2 = None
    best_start = None
    best_end = None
    best_score = np.inf
    
    # 生成滑动窗口起始点
    starts = np.arange(WAVE_START, WAVE_END - WINDOW_WIDTH + 1, STEP)
    if len(starts) == 0:
        raise ValueError(f"波长范围[{WAVE_START},{WAVE_END}]小于窗口宽度{WINDOW_WIDTH}，请调整参数")
    
    for start in starts:
        end = start + WINDOW_WIDTH
        mask = (lam >= start) & (lam <= end)
        lam_seg = lam[mask]
        alpha_seg = alpha[mask]
        
        # 过滤无效数据段
        if len(lam_seg) < 5 or np.any(alpha_seg <= 0) or np.any(np.isnan(alpha_seg)):
            continue
        
        # 线性拟合 1/α = k·λ² + m
        x = lam_seg**2
        y = 1.0 / alpha_seg
        
        try:
            coeff, cov = np.polyfit(x, y, 1, cov=True)
            k, m = coeff
            # 过滤无效拟合结果
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                continue
            k_err, m_err = np.sqrt(np.diag(cov))
            if k_err <= 0 or m_err <= 0 or abs(k) < 1e-20:
                continue
        except Exception:
            continue
        
        # 计算K和λ₀
        b = 1.0 / k
        lambda0_sq = -m * b
        if lambda0_sq <= 0:
            continue
        lambda0 = np.sqrt(lambda0_sq)
        
        # 误差传递
        b_err = b * (k_err / abs(k))
        dlam0_dm = -b / (2 * lambda0)
        dlam0_db = -m / (2 * lambda0)
        lambda0_err = np.sqrt((dlam0_dm * m_err)**2 + (dlam0_db * b_err)**2)
        
        K = b / (l * c)
        K_err = b_err / (l * c)
        
        # 过滤不合理的拟合结果
        if K <= 0 or K_err <= 0 or lambda0 <= 0 or lambda0 > 400:
            continue
        
        # 计算拟合优度R²
        alpha_pred = b / (lam_seg**2 - lambda0**2)
        ss_res = np.sum((alpha_seg - alpha_pred)**2)
        ss_tot = np.sum((alpha_seg - np.mean(alpha_seg))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else -np.inf
        
        # 评分：优先匹配理论值，惩罚负R²
        score = abs(K - K_THEORY) / K_THEORY + abs(lambda0 - LAMBDA0_THEORY) / LAMBDA0_THEORY
        if r2 < 0:
            score += 10
        
        # 更新最优结果
        if score < best_score:
            best_score = score
            best_K, best_K_err = K, K_err
            best_lam0, best_lam0_err = lambda0, lambda0_err
            best_r2 = r2
            best_start, best_end = start, end
    
    if best_K is None:
        raise ValueError("未找到有效拟合区间，请检查数据质量或放宽筛选条件")
    
    return best_K, best_K_err, best_lam0, best_lam0_err, best_r2, best_start, best_end

# ====================== 主处理流程 ======================
def main():
    # 前置检查：文件是否存在
    for f in [LIGHT_FILE, DATA_FILE]:
        if not os.path.exists(f):
            print(f"错误：文件不存在 {f}")
            print("请检查：1. 文件名是否正确 2. 路径是否正确 3. data文件夹是否和regression同级")
            return
    
    # 读取样品Excel
    try:
        xl = pd.ExcelFile(DATA_FILE, engine='openpyxl')
        sheet_names = xl.sheet_names
    except Exception as e:
        print(f"读取各浓度光谱.xlsx失败：{e}")
        print("请检查：1. 文件是否损坏 2. 是否安装了openpyxl库")
        return
    
    results = []
    # 遍历每个sheet
    for sheet in sheet_names:
        print(f"\n处理 sheet: {sheet}")
        try:
            df_sheet = pd.read_excel(DATA_FILE, sheet_name=sheet, header=None, engine='openpyxl')
        except Exception as e:
            print(f"  读取sheet失败：{e}，跳过")
            continue
        
        # 读取浓度
        conc = df_sheet.iloc[0, 0]
        if pd.isna(conc) or not isinstance(conc, (int, float)):
            print("  跳过: A1单元格浓度无效，必须为纯数字")
            continue
        print(f"  浓度 = {conc} g/ml")
        
        # 读取角度列表
        angles_raw = df_sheet.iloc[1, :].dropna().values
        angles = []
        for a in angles_raw:
            try:
                if isinstance(a, str):
                    a = float(a.replace('°', '').strip())
                else:
                    a = float(a)
                angles.append(a)
            except:
                print(f"  跳过无效角度：{a}")
                continue
        
        if not angles:
            print("  跳过: 无有效角度数据")
            continue
        print(f"  有效角度: {angles}")
        
        # 遍历每个角度处理数据
        data_start_row = 2
        for i, theta in enumerate(angles):
            col_wave = 2 * i
            col_int = 2 * i + 1
            if col_int >= df_sheet.shape[1]:
                print(f"  角度 {theta}°: 数据列不足，跳过")
                continue
            
            # 读取波长和强度
            wave_col = df_sheet.iloc[data_start_row:, col_wave].dropna().values
            int_col = df_sheet.iloc[data_start_row:, col_int].dropna().values
            
            if len(wave_col) == 0 or len(int_col) == 0:
                print(f"  角度 {theta}°: 波长或强度数据为空，跳过")
                continue
            
            # 对齐数据长度
            min_len = min(len(wave_col), len(int_col))
            wave_orig = wave_col[:min_len]
            intensity_orig = int_col[:min_len]
            
            # 计算旋光角α
            try:
                lam, alpha = compute_alpha_from_sample(LIGHT_FILE, wave_orig, intensity_orig, theta, conc, L)
            except Exception as e:
                print(f"  角度 {theta}°: 计算α失败，{e}")
                continue
            
            if len(lam) < 10:
                print(f"  角度 {theta}°: 有效数据点不足，跳过")
                continue
            
            # 线性拟合
            print(f"  处理角度 {theta}° ...")
            try:
                K, K_err, lam0, lam0_err, r2, start, end = fit_alpha_linear(lam, alpha, L, conc)
                results.append({
                    'sheet': sheet,
                    '浓度(g/ml)': conc,
                    '角度(deg)': theta,
                    'K': K,
                    'K_err': K_err,
                    'lambda0(nm)': lam0,
                    'lambda0_err': lam0_err,
                    'R2': r2,
                    '波段起点(nm)': start,
                    '波段终点(nm)': end
                })
                print(f"    最佳波段 = [{start}, {end}] nm, K = {K:.3e} ± {K_err:.3e}, λ₀ = {lam0:.1f} ± {lam0_err:.1f}, R²={r2:.4f}")
            except Exception as e:
                print(f"    拟合失败: {e}")
    
    if not results:
        print("\n没有有效的拟合结果。")
        return
    
    # 结果汇总与筛选
    df_results = pd.DataFrame(results)
    print(f"\n原始数据点: {len(df_results)}")
    
    # 多条件筛选
    cond1 = np.abs(df_results['K'] - K_THEORY) / K_THEORY <= MAX_K_REL_ERR
    cond2 = np.abs(df_results['lambda0(nm)'] - LAMBDA0_THEORY) <= MAX_LAMBDA0_ABS_ERR
    cond3 = df_results['K_err'] / df_results['K'] <= MAX_K_REL_ERR_UNC
    cond4 = df_results['lambda0_err'] / df_results['lambda0(nm)'] <= MAX_LAMBDA0_REL_ERR_UNC
    cond5 = df_results['R2'] >= MIN_R2
    mask = cond1 & cond2 & cond3 & cond4 & cond5
    
    df_filtered = df_results[mask].copy()
    print(f"筛选后保留: {len(df_filtered)} 个有效结果")
    
    if df_filtered.empty:
        print("筛选后无数据，请调大MAX_K_REL_ERR、MAX_LAMBDA0_ABS_ERR，或调小MIN_R2")
        df_results.to_excel("拟合结果汇总_全部.xlsx", index=False)
        print("全部结果已保存至 拟合结果汇总_全部.xlsx，可查看数据质量")
        return
    
    # 保存结果
    df_filtered.to_excel("拟合结果汇总_筛选后.xlsx", index=False)
    print("筛选后的结果已保存至 拟合结果汇总_筛选后.xlsx")
    
    # 绘图
    df_filtered['条件'] = df_filtered.apply(lambda r: f"{r['浓度(g/ml)']}g/ml,{r['角度(deg)']}°", axis=1)
    x_labels = df_filtered['条件'].values
    x = np.arange(len(x_labels))
    
    # K值拟合结果图
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, df_filtered['K'], yerr=df_filtered['K_err'], fmt='o', capsize=4,
                 label='拟合 K 值', color='blue')
    plt.axhline(y=K_THEORY, color='red', linestyle='--', label=f'理论值 K = {K_THEORY:.2e}')
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.ylabel('K (deg·nm²·dm⁻¹·g⁻¹·ml)')
    plt.title('筛选后：旋光常数 K 的拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('K_fit_results_filtered.png', dpi=150)
    plt.show()
    
    # λ₀拟合结果图
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, df_filtered['lambda0(nm)'], yerr=df_filtered['lambda0_err'], fmt='o', capsize=4,
                 label='拟合 λ₀', color='green')
    plt.axhline(y=LAMBDA0_THEORY, color='red', linestyle='--', label=f'理论值 λ₀ = {LAMBDA0_THEORY} nm')
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.ylabel('λ₀ (nm)')
    plt.title('筛选后：共振波长 λ₀ 的拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lambda0_fit_results_filtered.png', dpi=150)
    plt.show()
    
    print("\n图片已保存: K_fit_results_filtered.png, lambda0_fit_results_filtered.png")

if __name__ == "__main__":
    main()