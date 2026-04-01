在原有优化方法的基础上，我已将**不确定度计算**整合进“参数提取与误差分析”部分，其余内容保持不变。修改后的完整方法如下：

---

### 优化后的数据处理与参数拟合方法（含不确定度计算）

#### 1. 光谱比值与插值
设光源光谱为 \(I_0(\lambda)\)，透过系统后的光谱为 \(I(\lambda)\)。由于两光谱的波长采样点可能不一致，首先对 \(I_0(\lambda)\) 进行**线性插值**，使其在 \(I(\lambda)\) 的每个波长点 \(\lambda_i\) 上均有对应值。然后计算**透过率比值**：
\[
R(\lambda_i) = \frac{I(\lambda_i)}{I_0(\lambda_i)},
\]
为减小随机噪声，可对 \(R(\lambda)\) 进行**平滑处理**（如Savitzky‑Golay滤波或滑动平均），以稳定后续拟合。

#### 2. 归一化与反余弦变换的稳健处理
根据系统透过率模型：
\[
T(\lambda) = \cos^2\bigl(\alpha(\lambda) - \theta_0\bigr),
\]
其中 \(\alpha(\lambda) = l c \dfrac{K}{\lambda^2 - \lambda_0^2}\)，\(\theta_0\) 为起偏器与检偏器透振方向的夹角（已知）。在实验条件下，\(R(\lambda)\) 与 \(T(\lambda)\) 成比例（比例因子为探测系统增益 \(G\)），即：
\[
R(\lambda) = G \cdot \cos^2\bigl(\alpha(\lambda) - \theta_0\bigr).
\]

为避免直接除以最大值引入的误差，可采用**带参数的非线性拟合**同时求解 \(G\)、\(\alpha(\lambda)\) 中的参数。若仍希望先提取 \(\alpha(\lambda)\)，则改进步骤如下：

1. **归一化**：取 \(R(\lambda)\) 的最大值 \(R_{\max}\)，得归一化透过率：
   \[
   T_{\text{norm}}(\lambda) = \frac{R(\lambda)}{R_{\max}}.
   \]
2. **反余弦**：计算 \(\phi(\lambda) = \arccos\left(\sqrt{T_{\text{norm}}(\lambda)}\right)\)，此时 \(\phi(\lambda) = |\alpha(\lambda) - \theta_0|\)（取正值）。
3. **符号确定**：在波长范围内，\(\alpha(\lambda)\) 是单调递减函数（因 \(\lambda^2\) 增大）。选择某个波长点（如最大或最小波长）的理论值或实验趋势判断符号，使得 \(\alpha(\lambda) = \theta_0 \pm \phi(\lambda)\) 保持单调。通常可设定 \(\alpha(\lambda) = \theta_0 + \phi(\lambda)\) 或 \(\theta_0 - \phi(\lambda)\)，并根据整个波长范围内 \(\alpha(\lambda)\) 是否单调递减来修正符号。

#### 3. 拟合模型的改进与奇异值处理
由 \(\alpha(\lambda) = \dfrac{b}{\lambda^2 - \lambda_0^2}\)（其中 \(b = K l c\)）。该模型在 \(\lambda = \lambda_0\) 处有奇点，但 \(\lambda_0\) 通常位于紫外区（~146 nm），远离可见光波段，因此拟合时需确保 \(\lambda_0^2 < \lambda_{\min}^2\)，并可使用**约束优化**（如设置 \(\lambda_0\) 的上限）。

为提高拟合稳定性，建议采用**非线性最小二乘**，直接对以下方程进行拟合：
\[
\alpha(\lambda_i) = \frac{b}{\lambda_i^2 - \lambda_0^2},
\]
或等价地，将模型线性化：
\[
\frac{1}{\alpha(\lambda_i)} = \frac{\lambda_i^2}{b} - \frac{\lambda_0^2}{b}.
\]
即 \(y_i = 1/\alpha_i\) 与 \(x_i = \lambda_i^2\) 呈线性关系，斜率为 \(1/b\)，截距为 \(-\lambda_0^2/b\)。但需注意 \(\alpha(\lambda_i)\) 可能接近零导致 \(1/\alpha\) 发散，因此线性化方法仅适用于 \(\alpha\) 远大于测量误差的区域。更稳健的方式是使用**非线性最小二乘**，配合加权（如以 \(\alpha\) 的测量不确定度的倒数作为权重）。

#### 4. 参数提取与不确定度计算
通过非线性拟合得到参数估计值 \(\hat{b}\) 和 \(\hat{\lambda}_0\)，以及它们的**协方差矩阵** \(\mathbf{C}\)。拟合软件（如scipy.optimize.curve_fit）可直接输出参数的标准差 \(\sigma_b\)、\(\sigma_{\lambda_0}\) 及相关系数 \(\rho\)。

由 \(b = K l c\) 得：
\[
K = \frac{b}{l c}.
\]
考虑 \(l\)、\(c\) 的测量不确定度 \(\sigma_l\)、\(\sigma_c\)（通常由仪器精度或读数误差给出），利用**误差传递公式**计算 \(K\) 的综合不确定度 \(\sigma_K\)：
\[
\sigma_K = K \sqrt{ \left(\frac{\sigma_b}{b}\right)^2 + \left(\frac{\sigma_l}{l}\right)^2 + \left(\frac{\sigma_c}{c}\right)^2 + 2\rho\frac{\sigma_b\sigma_{\lambda_0}}{b\lambda_0} \cdot \frac{\partial \ln K}{\partial \lambda_0}? }
\]
但更严谨的做法是：由于 \(K\) 仅直接依赖于 \(b\)，而 \(b\) 与 \(\lambda_0\) 之间存在相关性，因此需通过**全微分**计算：
\[
\sigma_K^2 = \left(\frac{\partial K}{\partial b}\right)^2 \sigma_b^2 + \left(\frac{\partial K}{\partial l}\right)^2 \sigma_l^2 + \left(\frac{\partial K}{\partial c}\right)^2 \sigma_c^2 + 2\frac{\partial K}{\partial b}\frac{\partial K}{\partial l}\text{Cov}(b,l) + \cdots
\]
其中 \(\partial K/\partial b = 1/(l c)\)，\(\partial K/\partial l = -b/(l^2 c)\)，\(\partial K/\partial c = -b/(l c^2)\)。由于 \(l\)、\(c\) 与 \(b\) 相互独立（分别由长度测量、浓度配制引入），协方差项 \(\text{Cov}(b,l)=\text{Cov}(b,c)=0\)。因此：
\[
\sigma_K = \frac{1}{l c} \sqrt{ \sigma_b^2 + \left(\frac{b}{l}\right)^2 \sigma_l^2 + \left(\frac{b}{c}\right)^2 \sigma_c^2 }.
\]
若拟合中同时估计了 \(\lambda_0\)，且 \(\lambda_0\) 与 \(b\) 存在相关性，则 \(\sigma_b\) 已经包含了该相关性带来的影响（由协方差矩阵给出），无需额外引入交叉项。

最终结果表示为：
\[
K = \hat{K} \pm \sigma_K \quad (\text{单位}).
\]

#### 5. 拟合质量评估
- 计算决定系数 \(R^2\) 或调整 \(R^2\)。
- 绘制残差图，检查残差是否随机分布。
- 对比拟合得到的 \(\lambda_0\) 与文献值（146 nm）的接近程度，验证实验可靠性。

---

### 不确定度计算要点总结
| 不确定度来源                  | 处理方法                                                    |
| ----------------------------- | ----------------------------------------------------------- |
| 拟合参数 \(b\)、\(\lambda_0\) | 从协方差矩阵获取标准差 \(\sigma_b\)、\(\sigma_{\lambda_0}\) |
| 长度 \(l\)                    | 由旋光管标称精度或多次测量标准差给出 \(\sigma_l\)           |
| 浓度 \(c\)                    | 由称量质量、容量瓶体积及配制过程误差传递得到 \(\sigma_c\)   |
| 参数相关性                    | 通过误差传递公式中的协方差项（若相关）自动计入              |

---

此方法完整保留了原有的优化步骤，并明确加入了不确定度计算细节，可满足PPT中“数据处理与误差分析”部分的严谨性要求。