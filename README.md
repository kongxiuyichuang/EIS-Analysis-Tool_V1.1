# EIS Analysis Tool V1.1

**电化学阻抗谱数据处理与分析桌面工具**

面向 EIS 数据的一站式分析软件：从数据导入、Kramers-Kronig 校验、传输线模型（TLM）等效电路拟合，到弛豫/相位角/扩散时间分布反演（DRT / DCT / gDRT / DRT-DOP / DRT-DDT / DOP / DDT）与阻抗谱仿真，提供从宏观等效电路到微观弛豫分布的多维度分析能力。

完整使用说明见 [docs/manual_V1.1.md](docs/manual_V1.1.md)。

---

## 目录

1. [核心功能](#1-核心功能)
2. [方法选择指南](#2-方法选择指南)
3. [快速开始](#3-快速开始)
4. [测试](#4-测试)
5. [技术架构](#5-技术架构)
6. [特色亮点](#6-特色亮点)
7. [应用场景](#7-应用场景)
8. [依赖环境](#8-依赖环境)
9. [参考文献](#9-参考文献)

---

## 1. 核心功能

### 1.1 数据导入与 K-K 校验

支持 CSV、TXT、XLSX、XLS 等常见格式的三列阻抗数据（频率 $f$、实部 $Z'$、虚部 $Z''$）导入，内置频率筛选可截取目标频段。

Kramers-Kronig 校验基于 Schönleber 等人的 R-(RC)_M 串联等效电路方法，采用 Boukamp 加权的线性最小二乘，通过 μ 准则衡量数据一致性；并实现**多准则自动优化模式**——全范围扫描 M 值后结合曲率评分和 χ² 评分自动选择最优 RC 数量，避免传统 μ 截断法在 Warburg 型谱上选择过少 RC 元素的问题。

### 1.2 传输线模型（TLM）拟合

实现 Siroma 等人推导的完整 TLM 阻抗解析解，支持三种拓扑结构和八种边界条件：

| 拓扑 | 边界条件 | 数学特征 |
|:----|:---------|:---------|
| **Z-type** | P-Q、open-Q、open-open | z_A 与 z_C 对称 |
| **T-type** | P-Q、P-P symmetric、open-Q、open-open | z_A 为主导路径 |
| **E-type** | P-Q、open-Q、open-open | z_A 短路，退化为单层模型 |
| **Universal-type** | 通用八变量公式 | 通过 z_V/W/X/Y 自由组合所有边界条件 |

传播常数 β = √((z_A + z_C)/z_B)·L 与 α = √(z_A/z_B)·L（z_C = 0 时）统一描述多孔电极内部的电子/离子传输耦合行为。

### 1.3 元件编辑器与自定义电路

提供灵活的电化学电路定义语言——**CDC 语法**（Circuit Description Code）：

- **串联**：`A-B`；**并联**：`(AB)`；**嵌套**：任意深度括号组合

| 元件 | 阻抗公式 | 参数 |
|:----|:---------|:-----|
| **R**（电阻） | Z = R | `_R` |
| **C**（常相角元件 CPE） | Z = 1/(C(jω)^p) | `_C`, `_p` |
| **L**（电感） | Z = jωL | `_L` |
| **Wo**（有限空间扩散，频散） | Z = Rd·coth((jωτ)^p)/((jωτ)^p) | `_Rd`, `_tau`, `_p` |
| **Ws**（有限长度扩散，频散） | Z = Rd·tanh((jωτ)^p)/((jωτ)^p) | `_Rd`, `_tau`, `_p` |
| **W**（半无限 Warburg） | Z = Aw/√(jω) | `_Aw` |
| **H/Ha**（Havriliak-Negami） | 弛豫型经验公式 | `_dC/_R`, `_tau`, `_a`, `_b` |
| **G/Ga**（Gerischer） | 扩散-反应耦合模型 | `_Y/_R`, `_k/_tau`, `_n` |

Wo / Ws 引入频散参数 p（0~0.5，默认 0.5），将经典 Warburg 推广为带频散指数的一般形式，可描述非理想扩散。

### 1.4 CNLS 拟合

采用 Levenberg-Marquardt 算法的复非线性最小二乘，以 Boukamp 权重 $w_n=1/|Z_n|^2$ 加权。参数表支持独立设置初始值、上下界与固定标志；结果含 χ²、约化 χ²、R² 与归一化残差。

### 1.5 分布反演（7 种方法）

#### 1.5.1 DRT / DCT / gDRT（Tikhonov + QP）

均基于 PWL（分段线性）或 8 种 RBF 离散化 + Tikhonov 正则化，通过 cvxopt QP 求解器做非负约束二次规划：

- **DRT（阻抗空间弛豫时间分布）**：将阻抗谱分解为 RC 元件分布，采用 dlnτ 空间积分形式：
  $$Z(\omega) = R_0 + \frac{1}{j\omega C} + \int_{-\infty}^{\infty} \frac{g(\tau)}{1 + j\omega\tau} d\ln\tau$$
  其中 $g(\tau)=\tau\gamma(\tau)$ 为每十年时间常数的电阻密度（Ω/lnτ-decade），单 RC 回路满足 $\int g(\tau)d\ln\tau=R$。
- **DCT（容性时间分布）**：在导纳空间 $Y=1/Z$ 反演，其 Maxwell 并联结构可严格导出 FLW、FSW 与半无限 Warburg 的理论级数解，特别适合扩散主导数据。
- **gDRT（广义 DRT）**：同时处理容性（RC）与感性（RL）响应，按 $-\text{Im}(Z)$ 符号自动分离，并以零交点 $f_0$ 自适应确定两组 τ 范围。

支持 λ 自动选择（GCV / mGCV / L 曲线），统计面板显示实际使用的 λ。

#### 1.5.2 DRT-DOP / DRT-DDT / DOP / DDT（QPHB）

算法移植自 [hybrid-drt](https://github.com/jdhuang-csm/hybrid-drt)，采用 **QPHB 分层贝叶斯**迭代正则化。总模型为：

$$Z(\omega) = R_\infty + j\omega L + \frac{C_{\text{inv}}}{j\omega} + Z_{\text{DRT}}(\omega) + Z_{\text{DOP}}(\omega) + Z_{\text{DDT}}(\omega)$$

| 方法 | 求解 | 输出 |
|:-----|:-----|:-----|
| **DRT-DOP** | 联合求解 | $g(\tau)$ + $\rho(\nu)$ |
| **DRT-DDT** | 联合求解 | $g(\tau)$ + $q(\tau_D)$ |
| **DOP** | 纯相位角分布 | $\rho(\nu)=\int\rho(\nu)(j\omega)^\nu d\nu$ |
| **DDT** | 纯扩散时间分布 | $q(\tau_D)=\sum_n q_n Z_{\text{kernel}}(\omega,\tau_{D,n})$ |

- **DDT 核**（$u=(j\omega\tau_D)^p$）：1 = 平面开放（透射）$\coth(u)/u$；2 = 圆柱阻塞 $I_0(u)/(uI_1(u))$；3 = 平面短路 $\tanh(u)/u$
- **DDT 三种模式**：标准 1D（固定 p=0.5）/ 1D 标量 p 优化（[0.05, 0.5]）/ **2D 异常 DDT**（$q(\tau_D,p)$ 热图，默认 $p=\mathrm{linspace}(0.1,0.5,50)$）；`fit_ddt_p` 与 `fit_ddt_2d` 互斥
- **主要参数**（GUI 默认，与 facade 一致）：`DRT_l2_λ0=140`、`DOP_l2_λ0=10`、`DDT_l2_λ0=1.0`、`rp_scale=50`、`max_iter=30`

> **注意**：QPHB 的 `l2_λ0` 是 0/1/2 阶惩罚之和的全局尺度（叠加 `derivative_weights` 与自适应 $\rho_k$、局部 $s_k$），与 legacy 的单一二阶 λ **无精确倍率关系**。

### 1.6 阻抗谱仿真

支持模型选择、参数输入与频率扫描全流程仿真，实时观察 Nyquist、Bode、相位图与实虚部谱变化，便于理解 TLM 行为与验证拟合结果。

---

## 2. 方法选择指南

| 方法 | 适用 | 不适用 |
|:-----|:-----|:-------|
| **DRT** | 容性主导 EIS（电池、腐蚀等） | 强电感主导 |
| **DCT** | 扩散主导、高频低阻抗 | 极小阻抗噪声导致导纳爆炸 |
| **gDRT** | 同时含电感区 + 电容区 | 单侧频带过窄 |
| **DRT-DOP / DOP** | CPE、分数阶、相位角连续分布 | 仅需离散 RC 峰时用 DRT 即可 |
| **DRT-DDT / DDT** | 有限/半无限/多孔电极扩散 | 无明显扩散特征 |
| **2D DDT** | 异常扩散（p ≠ 0.5） | 数据点过少、噪声大 |

---

## 3. 快速开始

```bash
# 1) 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2) 启动
python main.py
```

已构建的可执行程序位于 `dist/EIS_Analysis_Tool_V1.0/`（PyInstaller 打包，含 `_internal` 运行库）。

---

## 4. 测试

```bash
python tests/quick_test.py        # 一键回归（TLM/DRT/DCT/gDRT/DOP/DDT 求解与界面冒烟）
python tests/test_core.py         # 核心数值测试
```

---

## 5. 技术架构

| 层次 | 技术选型 |
|:----|:---------|
| **GUI 框架** | PyQt5（QMainWindow + QTabWidget + QDock + QScrollArea） |
| **数值计算** | numpy、scipy、cvxopt |
| **数据可视化** | matplotlib（FigureCanvasQTAgg + NavigationToolbar2QT） |
| **数据 I/O** | pandas、openpyxl |

```
EIS_Analysis_Tool/
├── main.py                    # 程序入口
├── requirements.txt           # 依赖清单
├── gui/                       # 图形界面层
│   ├── main_window.py         # 主窗口（4 标签页）
│   ├── tabs/                  # data_tab / fit_tab / drt_dct_tab / simulation_tab
│   ├── docks/                 # 导入、筛选、K-K、模型选择、参数、绘图、统计面板
│   └── widgets/               # 元件编辑器、参数表等复用控件
├── core/                      # 计算核心
│   ├── drt.py / dct.py / gdrt.py     # legacy DRT/DCT/gDRT（Tikhonov + QP）
│   ├── lambda_selection.py    # GCV / mGCV / L 曲线自动选 λ
│   ├── kramers_kronig.py / kk_optimized.py  # K-K 校验与自动优化
│   ├── rbf.py / basis.py      # 基函数与离散化
│   ├── component_parser.py / component_impedance.py  # CDC 语法与元件阻抗
│   ├── fitting_engine.py      # CNLS 拟合引擎
│   ├── models/                # TLM 模型（Z/T/E/Universal 型）
│   └── hybdrt/                # DRT-DOP/DDT 求解（QPHB）
│       ├── facade.py          # 统一入口 solve_drt_dop_ddt
│       ├── models/            # drt1d / drtbase / qphb
│       ├── matrices/          # basis / ddt / phasance / mat1d
│       ├── preprocessing.py   # 网格与预处理
│       ├── plotting.py        # 可视化
│       └── _mitlef / utils / filters / dataload
├── resources/styles.qss       # 界面样式
├── tests/                     # 测试用例（含示例 CSV）
├── docs/manual.md             # 完整使用手册
└── dist/                      # PyInstaller 打包产物（可选）
```

---

## 6. 特色亮点

### 6.1 频散 Warburg 元件

Wo / Ws 推广为带频散指数 p（0~0.5）的一般形式，能更灵活地拟合非理想扩散行为。

### 6.2 dlnτ 空间的 DRT 实现

采用形式 B（dlnτ 空间）离散化，$g(\tau)$ 的谱峰面积直接对应 RC 电阻，结果解读直观。

### 6.3 自动 RL/RC 分离（gDRT）

按 $-\text{Im}(Z)$ 符号自动定位零交点 $f_0$，自适应确定两组 τ 范围，无需手动指定。

### 6.4 多准则 K-K 自动优化

μ 过滤 + 曲率评分 + χ² 评分的多准则算法，对分布型谱获得更稳健的 RC 数量选择。

### 6.5 QPHB 分层贝叶斯 DRT-DOP/DDT

将弛豫时间（DRT）、相位角（DOP）、扩散时间（DDT）分布统一到分层贝叶斯框架联合反演，支持纯分量与 2D 异常扩散分析。

---

## 7. 应用场景

- **锂离子电池**：多孔电极阻抗解析、SEI 膜表征、电荷转移与扩散过程分离
- **超级电容器**：多孔碳电极的传输线建模、频率响应分析
- **燃料电池**：气体扩散层与催化层的阻抗解析
- **腐蚀电化学**：涂层阻抗谱与 DRT 分析、弛豫过程辨识
- **固态电解质**：界面阻抗与体相阻抗的 DRT 分离
- **扩散/CPE 体系**：DOP / DDT 相位角与扩散时间分布分析

---

## 8. 依赖环境

| 依赖 | 用途 |
|:-----|:-----|
| Python 3.10+ | 运行环境 |
| PyQt5 | 图形界面 |
| numpy, scipy | 数值计算 |
| matplotlib | 数据可视化 |
| cvxopt | 二次规划求解（QP） |
| pandas, openpyxl | 数据 I/O |

---

## 9. 参考文献

1. Schönleber, M., Klotz, D., & Ivers-Tiffée, E. (2014). A method for improving the robustness of linear Kramers-Kronig validity tests. *Electrochimica Acta*, 131, 20-27.
2. Boukamp, B. A. (1995). A linear Kronig-Kramers transform test for immittance data validation. *Journal of the Electrochemical Society*, 142(6), 1885.
3. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution. *Electrochimica Acta*, 184, 483-499.
4. Ciucci, F., & Chen, C. (2015). Analysis of EIS data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. *Electrochimica Acta*, 167, 439-454.
5. Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics*, 2(2), 164-168.
6. Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.
7. Siroma, Z., et al. (2015). Mathematical solutions of comprehensive variations of a transmission-line model of the theoretical impedance of porous electrodes. *Electrochimica Acta*, 160, 313-322.
8. Plank, C., et al. (2024). A review on the distribution of relaxation times analysis. *Journal of Power Sources*, 594, 233845.
9. Py, B., Maradesa, A., & Ciucci, F. (2023). Theoretical and Experimental Study of the Distribution of Capacitive Times. *arXiv preprint*.
10. Danzer, M. A. (2019). Generalized Distribution of Relaxation Times Analysis for the Characterization of Impedance Spectra. *Batteries*, 5, 53.
11. Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. *Electrochimica Acta*, 247, 1117-1129.
12. Maradesa, A., et al. (2023). Selecting the regularization parameter in the distribution of relaxation times. *Journal of the Electrochemical Society*, 170, 030502.
13. Vogelsang, J., & Schiller, C. A. (2000). A new approach for the evaluation of impedance data by means of the Kramers-Kronig transformation. *Electrochimica Acta*, 45(22-23), 3837-3845.

---

*详细操作、数学原理与故障排除请参阅 [docs/manual.md](docs/manual.md)。*
