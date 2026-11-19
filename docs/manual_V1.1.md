# EIS Analysis Tool V1.0  使用手册

## 目录

1. [引言](#1-引言)
2. [快速入门](#2-快速入门)
3. [Data 标签页](#3-data-标签页)
4. [Fit 标签页](#4-fit-标签页)
5. [DRT-DCT 标签页](#5-drt-dct-标签页)
6. [Simulation 标签页](#6-simulation-标签页)
7. [常见问题与故障排除](#7-常见问题与故障排除)
8. [参考文献](#8-参考文献)

---

## 1. 引言

### 1.1 软件概述

EIS Analysis Tool 是一款面向电化学阻抗谱（EIS）数据处理与分析的桌面应用程序。它提供了从数据导入、Kramers-Kronig 校验、传输线模型（TLM）电路、等效电路拟合、弛豫/相位角/扩散时间分布（DRT/DCT/gDRT/DOP/DDT）反演到阻抗谱仿真的完整工作流。

#### 软件特色

- **复杂边界的传输线模型**：支持 Z 型、T 型、E 型三种拓扑结构和 P-Q、P-P、O-O 等八种边界条件，以及 Universal-type 通用公式，可覆盖所有多孔电极阻抗场景（Siroma et al. [7]）
- **广义 DRT（gDRT）**：在传统 DRT 基础上引入感性核函数，同时处理阻抗谱中的电感和电容响应，利用 $-\text{Im}(Z)$ 符号自动分离 RL 和 RC 分布
- **电容时间分布分析（DCT）**：在导纳空间进行 DRT 反演，适用于高频阻抗较小的数据
- **DRT-DOP / DRT-DDT（QPHB）**：基于分层贝叶斯（QPHB）的联合反演，可同时提取弛豫时间分布（DRT）、相位角分布（DOP）与扩散时间分布（DDT）；支持纯 DOP / 纯 DDT
- **自适应 τ 网格**：gDRT 中根据 $-\text{Im}(Z)$ 符号分布自动确定 τ 范围，确保 $f_0$ 处不扩展
- **自动 K-K 优化**：全范围扫描 + 多准则评分自动选择最优 RC 数量
- **自定义电路拟合**：通过元件编辑器自由组合电路结构，支持 R、C、L、Wo、Ws、W、G、Ga、H、Ha 等多种基本阻抗元件

### 1.2 适用场景

- **电池 EIS 数据分析**：对称电池、全电池、三电极体系的阻抗谱解析
- **传输线模型拟合**：多孔电极、固态电解质界面层的传输线建模
- **DRT / DCT / gDRT 分析**：弛豫时间分布反演，辨识电化学过程的特征时间常数
- **DOP / DDT 分析**：相位角分布与扩散时间分布（含 2D 异常扩散），适用于 CPE/Warburg/有限扩散等过程
- **阻抗谱仿真**：通过调整模型参数观察阻抗谱变化规律

### 1.3 依赖环境

| 依赖 | 用途 |
|:-----|:-----|
| Python 3.10+ | 运行环境 |
| PyQt5 | 图形界面 |
| numpy, scipy | 数值计算 |
| matplotlib | 数据可视化 |
| cvxopt | 二次规划求解（QP） |
| pandas, openpyxl | 数据 I/O |

---

## 2. 快速入门

### 2.1 数据格式要求

支持的文件格式：CSV、TXT、XLSX、XLS。

数据需包含三列，按以下顺序排列：

| 列 | 内容 | 单位 |
|:---|:-----|:-----|
| 第 1 列 | 频率 $f$ | Hz |
| 第 2 列 | 实部 $Z'$ | $\Omega$ |
| 第 3 列 | 虚部 $Z''$ | $\Omega$ |

频率建议从高到低排列。软件会自动排序。

### 2.2 典型工作流

```
导入 EIS 数据 (CSV/TXT/Excel)
       ↓
  频率筛选 (去除噪声频段)
       ↓
  K-K 校验 (验证数据一致性) ───→ 不合格则返回检查数据
       ↓
   ┌──────────┬──────────────────────────────────┬────────────┐
   ↓          ↓                                  ↓            ↓
 电路拟合   DRT / DCT / gDRT              DRT-DOP / DRT-DDT   仿真
  (Fit)    (Tikhonov + QP)               DOP / DDT (QPHB)  (Simulation)
   ↓          ↓                                  ↓
 等效电路   γ(τ) / RL+RC 分布              g(τ)+ρ(ν)/q(τ_D)
   ↓          ↓                                  ↓
              导出结果 (Excel)
```

### 2.3 界面总览

主窗口包含 4 个顶部标签页：

| 标签页 | 功能 |
|:-------|:-----|
| ① Data | 数据导入、频率筛选、K-K 校验、原始/K-K 可视化、版权信息 |
| ② Fit | 传输线模型选择、元件编辑、参数配置、CNLS 拟合、拟合结果可视化 |
| ③ DRT-DCT | 七种分布方法（DRT/DCT/gDRT/DRT-DOP/DRT-DDT/DOP/DDT）、谱图/Nyquist/残差、结果统计与导出 |
| ④ Simulation | 模型选择、参数输入、频率扫描、阻抗谱计算与可视化 |

每个标签页采用左右分栏布局：左侧为控制面板（可滚动），右侧为绘图区。

---

## 3. Data 标签页

### 3.1 数据导入

点击「导入数据」按钮，选择文件。支持自动识别分隔符。导入后将在信息标签中显示文件名、数据点数和频率范围。

### 3.2 频率筛选

通过拖拽 `f_min` 和 `f_max` 的滑块或输入数值，可以截取目标频段。筛选后的数据将作为后续 Fit 和 DRT-DCT 分析的数据源。

### 3.3 Kramers-Kronig 校验

#### 数学原理

K-K 校验（Schönleber et al. [1]）通过拟合串联 $R\text{-}(RC)_M$ 等效电路来验证 EIS 数据是否满足 Kramers-Kronig 关系：

$$Z(\omega) = R_0 + \sum_{k=1}^{M} \frac{R_k}{1 + j\omega\tau_k} + \frac{1}{j\omega C} + j\omega L \tag{1}$$

采用 Boukamp 权重 [2] 的加权最小二乘：

$$\chi^2 = \sum_{n=1}^{N} \frac{|\hat{Z}_n - Z_n|^2}{|Z_n|^2} \tag{2}$$

将问题转化为线性方程组 $A\mathbf{x} = \mathbf{b}$ 求解。

#### μ 准则

μ 值衡量拟合中物理意义的合理性：

$$\mu = 1 - \frac{\sum |R_k^{\text{neg}}|}{\sum |R_k^{\text{pos}}|} \tag{3}$$

其中 $R_k^{\text{pos}}$ 为正的 $R_k$ 值之和，$R_k^{\text{neg}}$ 为负的 $R_k$ 值之和。$\mu$ 越接近 1 表示电阻值越符合物理实际（全部为正）。一般以 $\mu > 0.85$ 为合格。

#### 自动优化（Auto-Optimized）模式

K-K 校验的精度受 RC 元素数量 $M$ 的选择影响。本软件支持**自动优化模式**，该模式不依赖单一的 $\mu$ 截断准则，而是通过多准则评分确定最优 $M$（Plank et al. [8]）：

1. **全范围扫描**：对 $M = 1$ 到 $M_{\max}$ 分别执行 K-K 拟合
2. **$\mu$ 过滤**：保留 $\mu > \mu_{\text{cutoff}}$ 的拟合结果
3. **曲率评分**：对每个候选 $M$ 计算其 $\gamma(\tau)$ 分布的曲率特征
4. **最优选择**：结合曲率准则和 $\chi^2$ 评分选择最佳 $M$

该模式的优势在于避免 $\mu$ 值随 $M$ 增加而单调下降时选择过小的 $M$，对分布型阻抗谱（如 Warburg 型）可获得更合理的 RC 数量。

#### 结果解读

- **RC 数量**：拟合所需的 RC 元件数。数量越多说明阻抗谱越复杂（分布型越强）
- **μ 值**：拟合物理一致性指标，$\mu > 0.85$ 表示数据通过 K-K 校验
- **伪卡方 $\chi^2$**：拟合质量，越小越好
- **残差统计**：归一化残差 $\Delta/\lvert Z \rvert$ 的最大百分比，反映模型与数据的偏差

---

## 4. Fit 标签页

### 4.1 传输线模型 (TLM)

传输线模型（Siroma et al. [7]）将多孔电极内部的复杂物理化学过程简化为由电子电阻、离子电阻和界面阻抗组成的等效电路网络。软件支持三种拓扑结构和八种边界条件。

#### 4.1.1 基本元器件定义

TLM 模型中的基本元器件包括：

| 符号 | 名称 | 阻抗公式 $Z(\omega)$ | 参数命名 |
|:----:|:-----|:--------------------|:---------|
| **R** | 电阻 | $Z_R = R$ | `_R` |
| **C** | 常相角元件 (CPE) | $\displaystyle Z_C = \frac{1}{C(j\omega)^p}$ | `_C`, `_p` |
| **L** | 电感 | $Z_L = j\omega L$ | `_L` |
| **Wo** | 有限空间扩散（反射边界） | $\displaystyle Z_{\text{Wo}} = R_d\frac{\coth((j\omega\tau_d)^p)}{(j\omega\tau_d)^p}$ | `_Rd`, `_tau`, `_p` |
| **Ws** | 有限长度扩散（透射边界） | $\displaystyle Z_{\text{Ws}} = R_d\frac{\tanh((j\omega\tau_d)^p)}{(j\omega\tau_d)^p}$ | `_Rd`, `_tau`, `_p` |
| **W** | 半无限 Warburg 扩散 | $\displaystyle Z_W = \frac{A_W}{\sqrt{j\omega}}$ | `_Aw` |
| **H** | Havriliak-Negami 弛豫 | $\displaystyle Z_H = \frac{(1+(j\omega\tau)^a)^b}{j\omega\Delta C}$ | `_dC`, `_tau`, `_a`, `_b` |
| **Ha** | HN 替代形式 | $\displaystyle Z_{Ha} = \frac{R}{(1+(j\omega\tau)^a)^b}$ | `_R`, `_tau`, `_a`, `_b` |
| **G** | Gerischer | $\displaystyle Z_G = \frac{1}{Y(k+j\omega)^n}$ | `_Y`, `_k`, `_n` |
| **Ga** | Gerischer 替代形式 | $\displaystyle Z_{Ga} = \frac{R}{(1+j\omega\tau)^n}$ | `_R`, `_tau`, `_n` |

#### 4.1.2 传播常数

传输线模型的特征由传播常数 $\gamma$ 描述：

$$\beta = \sqrt{\frac{z_A + z_C}{z_B}} \cdot L \tag{4}$$

$$\alpha = \sqrt{\frac{z_A}{z_B}} \cdot L \quad (z_C = 0) \tag{5}$$

其中 $L$ 为电极厚度（cm），$z_A$、$z_B$、$z_C$ 为各支路单位长度的复阻抗。

#### 4.1.3 Z-type 阻抗公式

当使用 `"Z-type"` 拓扑时，总阻抗 $Z$ 的表达式中 $z_A$ 和 $z_C$ 对称。

| 边界条件 | 阻抗公式 |
|:---------|:---------|
| **P-Q** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + \frac{\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{(z_A^2+z_C^2)\cosh\beta+2z_A z_C+\left(\frac{z_C^2}{z_P}+\frac{z_A^2}{z_Q}\right)\sqrt{z_B}(z_A+z_C)\sinh\beta}{\left\{1+\frac{1}{z_P z_Q}z_B(z_A+z_C)\right\}\sinh\beta+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_B}(z_A+z_C)\cosh\beta}$ |
| **open-Q** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + \frac{\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{(z_A^2+z_C^2)\cosh\beta+2z_A z_C+\frac{1}{z_Q}z_A^2\sqrt{z_B}(z_A+z_C)\sinh\beta}{\sinh\beta+\frac{1}{z_Q}\sqrt{z_B}(z_A+z_C)\cosh\beta}$ |
| **open-open** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + \frac{\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{(z_A^2+z_C^2)\cosh\beta+2z_A z_C}{\sinh\beta}$ |
| **P-Q ($z_C=0$)** | $\displaystyle Z = \frac{\sqrt{z_A z_B}\cosh\alpha+\frac{1}{z_Q}z_A z_B\sinh\alpha}{\left\{1+\frac{1}{z_P z_Q}z_A z_B\right\}\sinh\alpha+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_A z_B}\cosh\alpha}$ |
| **open-Q ($z_C=0$)** | $\displaystyle Z = \frac{\sqrt{z_A z_B}\cosh\alpha+\frac{1}{z_Q}z_A z_B\sinh\alpha}{\sinh\alpha+\frac{1}{z_Q}\sqrt{z_A z_B}\cosh\alpha}$ |
| **open-open ($z_C=0$)** | $\displaystyle Z = \frac{\sqrt{z_A z_B}\cosh\alpha}{\sinh\alpha} = \sqrt{z_A z_B}\coth\alpha$ |

边界条件符号说明：
- **P**（可穿透, Permeable）：边界处无额外阻抗
- **Q**（阻塞, Blocking）：边界处有界面阻抗 $z_P$ 或 $z_Q$
- **O**（不可穿透, Open）：边界处开路
- **W**（Warburg）：边界处为扩散阻抗

#### 4.1.4 T-type 阻抗公式

当使用 `"T-type"` 拓扑时，$z_A$ 侧为主导路径。

| 边界条件 | 阻抗公式 |
|:---------|:---------|
| **P-Q** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + \frac{z_A^2\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{2(\cosh\beta-1)+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_B}(z_A+z_C)\sinh\beta}{\left\{1+\frac{1}{z_P z_Q}z_B(z_A+z_C)\right\}\sinh\beta+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_B}(z_A+z_C)\cosh\beta}$ |
| **P-P**（对称） | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + 2\cdot\frac{z_A^2\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{(\cosh\beta-1)+\frac{1}{z_P}\sqrt{z_B}(z_A+z_C)\sinh\beta}{\left\{1+\frac{1}{z_P^2}z_B(z_A+z_C)\right\}\sinh\beta+\frac{2}{z_P}\sqrt{z_B}(z_A+z_C)\cosh\beta}$ |
| **open-Q** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + \frac{z_A^2\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{2(\cosh\beta-1)+\frac{1}{z_Q}\sqrt{z_B}(z_A+z_C)\sinh\beta}{\sinh\beta+\frac{1}{z_Q}\sqrt{z_B}(z_A+z_C)\cosh\beta}$ |
| **open-open** | $\displaystyle Z = \frac{z_A z_C}{z_A+z_C}L + 2\cdot\frac{z_A^2\sqrt{z_B}}{(z_A+z_C)^{3/2}}\cdot\frac{\cosh\beta-1}{\sinh\beta}$ |
| **P-Q ($z_C=0$)** | $\displaystyle Z = \sqrt{z_A z_B}\cdot\frac{2(\cosh\alpha-1)+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_A z_B}\sinh\alpha}{\left\{1+\frac{1}{z_P z_Q}z_A z_B\right\}\sinh\alpha+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_A z_B}\cosh\alpha}$ |
| **P-P ($z_C=0$)** | $\displaystyle Z = \sqrt{z_A z_B}\cdot 2\cdot\frac{(\cosh\alpha-1)+\frac{1}{z_P}\sqrt{z_A z_B}\sinh\alpha}{\left\{1+\frac{1}{z_P^2}z_A z_B\right\}\sinh\alpha+\frac{2}{z_P}\sqrt{z_A z_B}\cosh\alpha}$ |
| **open-Q ($z_C=0$)** | $\displaystyle Z = \sqrt{z_A z_B}\cdot\frac{2(\cosh\alpha-1)+\frac{1}{z_Q}\sqrt{z_A z_B}\sinh\alpha}{\sinh\alpha+\frac{1}{z_Q}\sqrt{z_A z_B}\cosh\alpha}$ |
| **open-open ($z_C=0$)** | $\displaystyle Z = 2\sqrt{z_A z_B}\cdot\frac{\cosh\alpha-1}{\sinh\alpha}$ |

#### 4.1.5 E-type 阻抗公式

当使用 `"E-type"` 拓扑时，$z_A$ 被视为短路，模型退化为单层传输线。

| 边界条件 | 阻抗公式 |
|:---------|:---------|
| **P-Q** | $\displaystyle Z = \frac{\sqrt{z_B(z_A+z_C)}\cosh\beta+\frac{1}{z_Q}z_B(z_A+z_C)\sinh\beta}{\left\{1+\frac{1}{z_P z_Q}z_B(z_A+z_C)\right\}\sinh\beta+\left(\frac{1}{z_P}+\frac{1}{z_Q}\right)\sqrt{z_B(z_A+z_C)}\cosh\beta}$ |
| **open-Q** | $\displaystyle Z = \frac{\sqrt{z_B(z_A+z_C)}\cosh\beta+\frac{1}{z_Q}z_B(z_A+z_C)\sinh\beta}{\sinh\beta+\frac{1}{z_Q}\sqrt{z_B(z_A+z_C)}\cosh\beta}$ |
| **open-open** | $\displaystyle Z = \frac{\sqrt{z_B(z_A+z_C)}\cosh\beta}{\sinh\beta} = \sqrt{z_B(z_A+z_C)}\coth\beta$ |

#### 4.1.6 Universal-type 通用公式

此模式允许用户通过 $z_V$、$z_W$、$z_X$、$z_Y$ 四个额外边界阻抗自由组合，覆盖所有可能的边界条件配置（Siroma et al. [7]）：

$$Z = \frac{z_A z_C}{z_A+z_C} \cdot L + \frac{\sqrt{z_B}}{(z_A+z_C)^{3/2}} \cdot \frac{A\cosh\beta + B\sinh\beta + C}{D\cosh\beta + E\sinh\beta} \tag{6}$$

其中系数 $A,B,C,D,E$ 定义为：

$$\begin{aligned}
A &= \{Z_W(Z_X+Z_Y) + (Z_V+Z_W)Z_Y + Z_V Z_W + Z_X Z_Y\} z_A^2 \\
  &\quad + 2(Z_V Z_W + Z_X Z_Y) z_A z_C \\
  &\quad + \{Z_V(Z_X+Z_Y) + (Z_V+Z_W)Z_X + Z_V Z_W + Z_X Z_Y\} z_C^2
\end{aligned} \tag{7}$$

$$\begin{aligned}
B &= \sqrt{z_B(z_A+z_C)} \cdot \Big[ (Z_W+Z_Y)z_A^2 + \{Z_V Z_W(Z_X+Z_Y) \\
  &\quad + (Z_V+Z_W)Z_X Z_Y\} \frac{z_A+z_C}{z_B} + (Z_V+Z_X)z_C^2 \Big]
\end{aligned} \tag{8}$$

$$C = -2 Z_W Z_Y z_A^2 + 2 (Z_W Z_X + Z_V Z_Y) z_A z_C - 2 Z_V Z_X z_C^2 \tag{9}$$

$$D = (Z_V+Z_W+Z_X+Z_Y)\sqrt{z_B(z_A+z_C)} \tag{10}$$

$$E = z_B(z_A+z_C) + (Z_V+Z_W)(Z_X+Z_Y) \tag{11}$$

通过设定 $z_V,z_W,z_X,z_Y$ 为 $0$ 或 $\infty$，通用公式可以退化为上述所有 8 种边界条件。

#### 4.1.7 模式说明

##### 非法拉第模式（Non-Faradaic）

适用于双电层电容行为（如不含活性物质的碳电极或锂离子电池阻塞电极）：

$$z_B = \frac{1}{C_{dl,\text{porous}}(j\omega)^p} \tag{12}$$

##### 法拉第模式（Faradaic）

适用于存在电荷转移反应的场景：

$$z_B = \frac{R_{ct}}{1 + C_{dl,\text{porous}}(j\omega)^p R_{ct}} \tag{13}$$

其中 $C_{dl,\text{porous}}$ 为多孔电极双电层电容，$p$ 为 CPE 指数，$R_{ct}$ 为电荷转移电阻。

#### 4.1.8 物理变量对照表

| 变量 | 含义 | 典型单位 |
|:-----|:-----|:---------|
| $R_0$ | 欧姆电阻（集流体+电解液+接触电阻） | $\Omega$ |
| $L$ | 电极厚度 | cm |
| $z_A$ | 单位长度的电子阻抗（$z_A$ 支路） | $\Omega$/cm |
| $z_B$ | 单位长度的界面阻抗（$z_B$ 支路） | $\Omega\cdot$cm |
| $z_C$ | 单位长度的电子阻抗（$z_C$ 支路） | $\Omega$/cm |
| $z_P$ | 边界 A 的界面阻抗 | $\Omega$ |
| $z_Q$ | 边界 B 的界面阻抗 | $\Omega$ |

### 4.2 元件定义与参数配置

用户通过元件编辑器定义各支路（zA、zB、zC、zP、zQ、zV、zW、zX、zY）的电路表达式。软件支持自定义模式，用户可编写任意电路结构的表达式。

#### 4.2.1 基本元件语法

| 元件 | 语法 | 参数 | 示例 | 等效电路含义 |
|:----|:-----|:-----|:-----|:------------|
| 电阻 | `R` | `_R` | `R1` | $\text{R}_1$ |
| CPE | `C` | `_C`, `_p` | `C1` | $\text{CPE}_1$ |
| 电感 | `L` | `_L` | `L1` | $\text{L}_1$ |
| 有限空间扩散（频散） | `Wo` | `_Rd`, `_tau`, `_p` | `Wo1` | $\text{Wo}_1$ |
| 有限长度扩散（频散） | `Ws` | `_Rd`, `_tau`, `_p` | `Ws1` | $\text{Ws}_1$ |
| 半无限 Warburg | `W` | `_Aw` | `W1` | $\text{W}_1$ |
| Havriliak-Negami | `H` | `_dC`, `_tau`, `_a`, `_b` | `H1` | $\text{HN}_1$ |
| HN 替代形式 | `Ha` | `_R`, `_tau`, `_a`, `_b` | `Ha1` | $\text{Ha}_1$ |
| Gerischer | `G` | `_Y`, `_k`, `_n` | `G1` | $\text{G}_1$ |
| Gerischer 替代形式 | `Ga` | `_R`, `_tau`, `_n` | `Ga1` | $\text{Ga}_1$ |

#### 4.2.2 串并联语法规则

| 语法 | 含义 | 等效表达式 |
|:----|:-----|:-----------|
| `A-B` | **串联**：元件 A 与 B 串联 | $Z_{\text{total}} = Z_A + Z_B$ |
| `(AB)` | **并联**：元件 A 与 B 并联 | $\displaystyle Z_{\text{total}} = \frac{1}{1/Z_A + 1/Z_B}$ |
| `(ABC...)` | 多元件并联 | $\displaystyle Z_{\text{total}} = \frac{1}{1/Z_A + 1/Z_B + 1/Z_C + \cdots}$ |
| 括号嵌套 | 括号内先计算，再与外层组合 | 见下方案例 |

#### 4.2.3 案例

**案例 1：两个电阻串联**

```
zA = R1-R2
```
等效电路：$\text{R}_1$ — $\text{R}_2$（串联）
参数：`R1_R`, `R2_R`

**案例 2：一个电阻与一个 CPE 并联（RC 并联）**

```
zB = (R1C1)
```
等效电路：$\text{R}_1 \parallel \text{CPE}_1$
参数：`R1_R`, `C1_C`, `C1_p`

**案例 3：电阻与 RC 并联支路串联**

```
zA = R1-(R2C1)
```
等效电路：$\text{R}_1$ — ($\text{R}_2 \parallel \text{CPE}_1$)
参数：`R1_R`, `R2_R`, `C1_C`, `C1_p`

**案例 4：电阻与 RC 并联支路再并联**

```
zB = (R1(R2C1))
```
等效电路：$\text{R}_1 \parallel (\text{R}_2 \parallel \text{CPE}_1)$ = $\text{R}_1 \parallel \text{R}_2 \parallel \text{CPE}_1$
参数：`R1_R`, `R2_R`, `C1_C`, `C1_p`

**案例 5：含有 Warburg 扩散的 Randles 电路**

```
zB = R1-(C1(R2-Wo1))
```
等效电路：$\text{R}_1 — (\text{CPE}_1 \parallel (\text{R}_2 — \text{Wo}_1))$

| 层次 | 含义 |
|:----|:------|
| `Wo1` | 有限空间扩散阻抗 $Z_{\text{Wo}}$ |
| `R2-Wo1` | $R_2$ 与 $Wo_1$ 串联 |
| `(C1(R2-Wo1))` | $C_1$ 与上述串联体并联 |
| `R1-(C1(R2-Wo1))` | $R_1$ 与上述并联体串联 |

参数：`R1_R`, `R2_R`, `C1_C`, `C1_p`, `Wo1_Rd`, `Wo1_tau`, `Wo1_p`

#### 4.2.4 组件命名规则

- 组件名以字母开头，后跟数字：`R1`, `R2`, `C1`, `L1`, `Wo1`, `Ws1`, `W1`
- 组件类型由名称的首字母/前缀自动识别（`R`→ 电阻, `C`→ CPE, `L`→ 电感, `Wo`→ 有限空间扩散, `Ws`→ 有限长度扩散, `W`→ 半无限扩散, `H`→ Havriliak-Negami, `Ha`→ HN 替代形式, `G`→ Gerischer, `Ga`→ Gerischer 替代形式）
- 参数名自动生成：`组件名_参数后缀`（如 `R1_R`, `C1_C`, `C1_p`, `Wo1_Rd`, `Wo1_tau`, `Wo1_p`）

#### 4.2.5 自定义模式

选择 `"Custom"` 模型类型后，用户可以通过元件编辑器自由组合现有元件类型（R、C、L、Wo、Ws、W）构建任意等效电路结构，语法规则与 4.2.1~4.2.3 节完全一致：

```
zA = R1-(R2C1)
zB = (R3(Wo1C1))
```

自定义模式不引入新元件类型，仅允许用户在六个支路（zA、zB、zC、zP、zQ、zV、zW、zX、zY）中灵活分配不同的电路拓扑，无需受预设模板限制。

#### 4.2.6 参数配置表

参数表包含 5 列：

| 列 | 内容 | 说明 |
|:---|:-----|:-----|
| 参数名 | 自动生成，如 `R1_R` | 不可编辑 |
| 初始值 | 拟合的起始猜测 | 用户可调 |
| 下界 | 参数的最小允许值 | 用于约束拟合范围 |
| 上界 | 参数的最大允许值 | 用于约束拟合范围 |
| 固定 | 勾选后该参数不参与拟合 | 适用于已知值参数 |

### 4.3 CNLS 拟合算法

采用 Levenberg-Marquardt（LM）算法 [5, 6] 进行复非线性最小二乘拟合。

#### 目标函数

$$\min_{\mathbf{p}} \sum_{n=1}^{N} w_n \left| Z_n(\omega_n) - \hat{Z}_n(\omega_n, \mathbf{p}) \right|^2 \tag{14}$$

其中 $w_n$ 为 Boukamp 权重：

$$w_n = \frac{1}{|Z_n|^2} \tag{15}$$

#### Levenberg-Marquardt 更新

LM 算法（Levenberg [5], Marquardt [6]）结合了梯度下降法（远离最优解时）和高斯-牛顿法（接近最优解时）：

$$(J^T W J + \lambda \cdot \text{diag}(J^T W J)) \Delta\mathbf{p} = J^T W (\mathbf{Z} - \hat{\mathbf{Z}}) \tag{16}$$

其中 $J$ 为 Jacobian 矩阵（复数阻抗对各参数的偏导），$\lambda$ 为阻尼参数，在迭代过程中动态调整。

### 4.4 拟合结果解读

| 统计指标 | 定义 | 范围 | 评价标准 |
|:---------|:-----|:-----|:---------|
| $\chi^2$ 卡方 | $\sum w_n\lvert Z - \hat{Z} \rvert^2$ | $>0$ | 越小越好 |
| $\chi^2_\nu$ 约化卡方 | $\chi^2 / (N - M)$ | $>0$ | 接近 1 为最优 |
| $R^2$ 决定系数 | $1 - \sum w_n \lvert Z - \hat{Z}\rvert^2 / \sum w_n\lvert Z - \bar{Z}\rvert^2$ | $(-\infty, 1]$ | 接近 1 为最优 |
| **残差 $\Delta_{\text{re}}, \Delta_{\text{im}}$** | $(Z - \hat{Z})/\lvert Z\rvert$ | $\%$ | $\pm 5\%$ 以内合格 |

---

## 5. DRT-DCT 标签页

本标签页提供 **7 种** 分布反演方法，数据源为 Data 标签页筛选后的频率/阻抗。左侧按方法自动切换参数面板；右侧含「分布谱图」「残差分析」两个绘图页签。求解后可在统计区查看参数与残差，并导出 Excel。

### 5.1 方法总览与界面

| 方法 | 求解器 | 主要输出 | 参数面板 |
|:-----|:-------|:---------|:---------|
| **DRT** | Tikhonov + QP（cvxopt） | $g(\tau)$、$R_0$、可选 $C$ | Parameters set |
| **DCT** | 同上（导纳空间） | $g_{\text{DCT}}(\tau)$、$G_\infty$、$C_\infty$ | Parameters set |
| **gDRT** | 同上（RC+RL 双核） | $g_{\text{rc}}$、$g_{\text{rl}}$、$f_0$ | Parameters set |
| **DRT-DOP** | QPHB（hybrid-drt） | DRT + DOP | DRT-DOP/DDT parameters |
| **DRT-DDT** | QPHB | DRT + DDT | 同上 |
| **DOP** | QPHB | 纯 $\rho(\nu)$ | 同上（DRT 控件禁用） |
| **DDT** | QPHB | 纯 $q(\tau_D)$ 或 2D $q(\tau_D,p)$ | 同上（DRT 控件禁用） |

**界面行为：**

- 切换 **DRT/DCT/gDRT** ↔ **DOP/DDT 系列** 时，参数组可见性自动切换；用户已改参数**不会被重置**。
- 纯 DOP / 纯 DDT 时，DRT 专属控件（basis、ppd、extend、`DRT_l2_λ0`）禁用，统计/导出也不列出这些项。
- `fit_ddt_p` 与 `fit_ddt_2d` 在界面上互斥（勾选其一自动取消另一）。
- legacy 方法在 GCV/mGCV/L 曲线下，统计面板显示**实际使用的** `lambda`（`lambda_used`）。

---

### 5.2 DRT / DCT / gDRT（Tikhonov + QP）

#### 5.2.1 DRT（阻抗空间弛豫时间分布）

DRT（Wan et al. [3]）将阻抗谱分解为一系列 RC 元件的分布：

$$Z(\omega) = R_0 + \frac{1}{j\omega C} + \int_{-\infty}^{\infty} \frac{g(\tau)}{1 + j\omega\tau} d\ln\tau \tag{17}$$

##### DRT 的两种等价形式

**形式 A（$d\tau$ 空间）：**
$$Z(\omega) = R_0 + \frac{1}{j\omega C} + \int_{0}^{\infty} \frac{\gamma(\tau)}{1 + j\omega\tau} d\tau$$
$\gamma(\tau)$ 单位：$\Omega$/s

**形式 B（$d\ln\tau$ 空间）：**
$$Z(\omega) = R_0 + \frac{1}{j\omega C} + \int_{-\infty}^{\infty} \frac{g(\tau)}{1 + j\omega\tau} d\ln\tau$$
$g(\tau) = \tau \cdot \gamma(\tau)$，单位：$\Omega$/lnτ-decade

**本软件采用形式 B**。对单个 RC 回路，$\int g(\tau)\,d\ln\tau = R$。

##### PWL 离散化

$$\int_{-\infty}^{\infty} \frac{g(\tau)}{1 + j\omega\tau} d\ln\tau \approx \sum_{k=1}^{M} \frac{g_k}{1 + j\omega\tau_k} \Delta_k \tag{18}$$

$$\Delta_k = \frac{1}{2} \ln \frac{\tau_{k+1}}{\tau_{k-1}} \tag{19}$$

$$A_{\text{re}}[n,k] = \frac{1}{1 + (\omega_n\tau_k)^2} \cdot \Delta_k,\quad
A_{\text{im}}[n,k] = \frac{-\omega_n\tau_k}{1 + (\omega_n\tau_k)^2} \cdot \Delta_k \tag{20–22}$$

##### RBF 离散化

$$g(\ln\tau) = \sum_{m=1}^{M} x_m \phi_m(\ln\tau),\quad
\phi_m(\ln\tau) = \phi(\epsilon\,(\ln\tau - \ln\tau_m))$$

| 基函数 | 形式 $\phi(\epsilon x)$ |
|:------|:------------------------|
| Gaussian | $\exp(-(\epsilon x)^2)$ |
| C0 / C2 / C4 / C6 Matérn | 标准 Matérn 族 |
| Inverse Quadratic | $1/(1+(\epsilon x)^2)$ |
| Inverse Quadric | $1/\sqrt{1+(\epsilon x)^2}$ |
| Cauchy | $1/(1+\lvert\epsilon x\rvert)$ |

形状控制：

| 方式 | 公式 |
|:-----|:-----|
| FWHM 系数（默认） | $\epsilon$ 由 coeff 与网格间距换算，默认 coeff = 0.5 |
| 形状因子 | $\epsilon = \text{coeff}$ 直接指定 |

DCT 与 gDRT 同样支持 RBF：DCT 用导纳核；gDRT 的 RC 用阻抗核、RL 用导纳核。

##### 配点策略

| 策略 | 配点 | 说明 |
|:-----|:-----|:-----|
| **logspace 网格** | $\tau$ 对数均匀 $N_{\tau}$ 点，频率范围上下各扩 1  decade | 由 `Number of τ` 控制（默认 400） |
| **τ = 1/f（数据对齐，默认）** | $\tau=1/f$，点数 = $N_f$ | 与 pyDRTtools / pyimpspec 一致 |

选「τ = 1/f」时 `N_tau` 自动禁用。RBF 输出经更密 `tau_fine` 插值显示。

##### 权重方案

| 权重 | 公式 | 说明 |
|:-----|:-----|:-----|
| **Boukamp** | $w_n = 1/\lvert Z_n\rvert^2$ | 相对误差加权 |
| **无（不加权，默认）** | $w_n = 1$ | 与 pyimpspec/pyDRTtools 一致 |

##### 模型项与正则化阶数

- **含 1/(jωC) 项**（仅 DRT，默认勾选）：$R_0 + 1/(j\omega C) + \text{DRT}$；取消则为 $R_\infty + \text{DRT}$
- **导数阶数**：Tikhonov 罚项 2 阶（默认）或 1 阶

##### 正则化参数 λ 的选择

| 方式 | 说明 |
|:-----|:-----|
| **手动（固定）**（默认） | 使用 Fix λ 输入值 |
| **GCV / mGCV / L曲线** | 自动选 λ；统计中显示实际 `lambda` |

自动方式下 Fix λ 作为初值（钳制在 $[10^{-7},10^0]$）。

##### Tikhonov 正则化与 QP

$$\min_{\mathbf{g} \ge 0} \left\| \mathbf{A}\mathbf{g} - \mathbf{b} \right\|_W^2 + \lambda \| \mathbf{L}\mathbf{g} \|^2 \tag{23}$$

$$\min_{\mathbf{x}} \frac{1}{2}\mathbf{x}^T\mathbf{P}\mathbf{x} + \mathbf{q}^T\mathbf{x},\quad \mathbf{x}\ge 0 \tag{24}$$

$$\mathbf{P} = 2(\mathbf{A}^T\mathbf{W}^T\mathbf{W}\mathbf{A} + \lambda \mathbf{L}^T\mathbf{L}),\quad
\mathbf{q} = -2\mathbf{b}^T\mathbf{W}^T\mathbf{W}\mathbf{A} \tag{25–26}$$

采用 cvxopt primal-dual interior point 求解。

#### 5.2.2 DCT（容性时间分布）

DCT（Py et al. [9]）在导纳空间 $Y=1/Z$ 反演：

$$Y(\omega) = G_\infty + j\omega C_\infty + \int_{0}^{\infty} \frac{j\omega\tau}{1 + j\omega\tau} g_{DCT}(\tau) d\ln\tau \tag{27}$$

离散核：

$$A_{\text{re}}[n,k] = \frac{(\omega_n\tau_k)^2}{1 + (\omega_n\tau_k)^2}\Delta\ln\tau_k,\quad
A_{\text{im}}[n,k] = \frac{\omega_n\tau_k}{1 + (\omega_n\tau_k)^2}\Delta\ln\tau_k \tag{29–30}$$

**理论优势**：FLW / FSW / 半无限 Warburg 的导纳均可严格写成 DCT 的 Maxwell 并联结构，故 DCT 尤其适用于扩散主导数据。数值流程（离散 + Tikhonov + QP）与 DRT 相同。

#### 5.2.3 gDRT（广义 DRT）

gDRT（Danzer [10], Plank et al. [8]）同时处理 RC 与 RL：

$$Z(\omega) = R_0 + \frac{1}{j\omega C} + j\omega L_0
+ \int_{\tau_{\text{rc}}} \frac{g_{\text{rc}}(\tau)}{1 + j\omega\tau} d\ln\tau
+ \int_{\tau_{\text{rl}}} \frac{g_{\text{rl}}(\tau)\,(j\omega\tau)}{1 + j\omega\tau} d\ln\tau \tag{30}$$

- **RC 核**：$A_{\text{re}}=1/(1+(\omega\tau)^2)$，$A_{\text{im}}=-\omega\tau/(1+(\omega\tau)^2)$
- **RL 核**：$A_{\text{re}}=(\omega\tau)^2/(1+(\omega\tau)^2)$，$A_{\text{im}}=\omega\tau/(1+(\omega\tau)^2)$

零交点 $f_0$ 由 $\text{Im}(Z)$ 符号变化线性插值得到；$\tau_0=1/(2\pi f_0)$ 在谱图中以虚线标出。RC/RL 的 $\tau$ 范围按 $-\text{Im}(Z)$ 符号带自适应，靠近 $f_0$ 一端不扩展。

#### 5.2.4 legacy 共用参数速查

| 参数 | 典型范围 | 默认 | 说明 |
|:-----|:---------|:----:|:-----|
| Fix $\lambda$ | $10^{-5}\sim10^{-2}$ | $10^{-3}$ | 越大分布越平滑 |
| λ 选择 | 手动/GCV/mGCV/L曲线 | 手动 | 自动时统计显示实际 λ |
| Number of τ | $100\sim500$ | $400$ | 仅 logspace 策略有效 |
| 配点策略 | logspace / τ=1/f | **τ=1/f** | |
| Weight | Boukamp / 无 | **无** | |
| RBF type | PWL + 8 种 RBF | PWL | 非 PWL 时形状控件启用 |
| Derivative order | 1 / 2 | 2 | |
| 含 1/(jωC) | 开/关 | 开 | 仅 DRT |

---

### 5.3 DRT-DOP / DRT-DDT / DOP / DDT（QPHB）

算法移植自 [hybrid-drt](https://github.com/jdhuang-csm/hybrid-drt)，采用 **QPHB**（分层贝叶斯）迭代正则化，与 §5.2 的单一 $\lambda$·二阶 Tikhonov **结构不同**，数值上不可简单换算。

#### 5.3.1 总模型

$$Z(\omega) = R_\infty + j\omega L + \frac{C_{\text{inv}}}{j\omega}
+ Z_{\text{DRT}}(\omega) + Z_{\text{DOP}}(\omega) + Z_{\text{DDT}}(\omega)$$

| 分量 | 公式 | 说明 |
|:-----|:-----|:-----|
| DRT | $\int g(\tau)/(1+j\omega\tau)\,d\ln\tau$ | 基函数：gaussian / delta / Cole-Cole / pwl |
| DOP | $\int \rho(\nu)\,(j\omega)^\nu\,d\nu$ | 相位角（phasance）分布 |
| DDT | $\sum_n q_n Z_{\text{kernel}}(\omega,\tau_{D,n})$ | 扩散时间分布 |

方法开关：`DRT-DOP` = DRT+DOP；`DRT-DDT` = DRT+DDT；`DOP` / `DDT` 为单独分量。

#### 5.3.2 DDT 核与求解模式

核变量 $u=(j\omega\tau_D)^p$：

| 类型 | 边界 | 阻抗 |
|:----:|:-----|:-----|
| 1 | Reflective boundary planar | $\coth(u)/u$ |
| 2 | Reflective boundary cylindrical | $I_0(u)/(u\,I_1(u))$ |
| 3 | Transmissive boundary planar | $\tanh(u)/u$ |

| fit_ddt_p | fit_ddt_2d | 模式 |
|:---:|:---:|:-----|
| 否 | 否（默认） | 1D DDT，固定 $p=0.5$ |
| 是 | 否 | 1D DDT，标量 $p\in[0.05,0.5]$ 优化 |
| 否 | 是 | 2D 异常 DDT：$q(\tau_D,p)$，默认 $p=\mathrm{linspace}(0.1,0.5,50)$ |

#### 5.3.3 主要参数（默认与 GUI / facade 一致）

| 参数 | 默认 | 说明 |
|:-----|:----:|:-----|
| `fit_ohmic` | 是 | 拟合 $R_\infty$ |
| `fit_inductance` | 否 | 拟合 $j\omega L$ |
| `fit_capacitance` | 否 | 拟合 $C_{\text{inv}}/j\omega$ |
| `ddt_kernel_type` | 1 | DDT 核 1/2/3 |
| `fit_ddt_p` / `fit_ddt_2d` | 否 / 否 | 互斥 |
| `basis_tau_ddt` | 留空=自动 | 或 `np.logspace(-4,4,40)` |
| DRT basis / tau grid ppd / Extend decade | gaussian / 10 / 1 | 仅含 DRT 的方法 |
| **`DRT_l2_λ0`** | **140** | DRT 块 QPHB 全局尺度 |
| **`DDT_l2_λ0`** | **1.0** | DDT 块 |
| **`DOP_l2_λ0`** | **10** | DOP 块 |
| `rp_scale` | 50 | 数据缩放目标 Rp |
| `max_iter` | 30 | QPHB 外迭代上限 |

**关于 λ0 与 legacy λ：**  
QPHB 的 `l2_lambda_0` 是 0/1/2 阶惩罚之和的全局尺度，并叠加 `derivative_weights`、自适应 $\rho_k$ 与局部 $s_k$，与 legacy 的单一二阶 $\lambda$ **无精确倍率关系**。三者默认比约为 DRT:DOP:DDT = 140:10:1。

#### 5.3.4 绘图与导出

- **联合**（DRT-DOP / DRT-DDT）：顶行 DRT 占两列，DOP/DDT 占一列；底行 Nyquist + Re/Im。
- **纯 DOP / DDT**：顶行整行对应分布；2D DDT 为 $\tau_D$–$p$ 热图 + $p$ 边缘曲线。
- 导出 Excel：`Statistics`（参数与残差）、`experiment_Z`、`Reconstructed_Z`，以及 DRT/DOP/DDT 分布表（若有）。

---

### 5.4 结果解读

#### 分布谱图

- **DRT**：$g(\tau)$ vs $\tau$，峰位 ≈ 时间常数，峰高 ≈ 过程电阻
- **gDRT**：蓝 = RC，粉 = RL，灰虚线 = $\tau_0$
- **DOP**：$\rho(\nu)$ vs $\nu$；$\nu=-0.5$ 虚线对应理想 Warburg
- **DDT**：$q(\tau_D)$ 或 2D 热图

#### 重构与残差

Nyquist / Bode 对比实验与重构；归一化残差 $\Delta/\lvert Z\rvert$ 宜在 $\pm 5\%$ 内随机分布。

---

### 5.5 方法选择指南

| 方法 | 适用 | 不适用 |
|:-----|:-----|:-------|
| **DRT** | 容性主导 EIS | 强电感主导 |
| **DCT** | 扩散主导、高频低阻抗 | 极低阻抗噪声导致 $Y$ 爆炸 |
| **gDRT** | 同时含电感区+电容区 | 单侧频带过窄 |
| **DRT-DOP / DOP** | CPE、分数阶、相位角连续分布 | 仅需经典 RC 峰时可用 DRT |
| **DRT-DDT / DDT** | 有限/半无限扩散、多孔电极扩散 | 无扩散特征时不必启用 |
| **2D DDT** | 异常扩散（$p\neq 0.5$） | 数据点过少、噪声大 |

---

## 6. Simulation 标签页

### 6.1 模型选择

与 Fit 标签页共享相同的模型选择控件，包括：

- 拓扑类型：Z 型、P 型、T 型
- 边界条件：P-Q、P-P、O-O、O-P、W-P、W-O、P-W、O-W
- 特殊选项：短路模式、$zC = 0$ 模式

### 6.2 参数设置

参数表包含 3 列：`参数名 | 值 | 单位`。用户手动调整各参数值，点击「计算」后软件根据当前模型和参数计算阻抗谱。

### 6.3 频率扫描

频率范围和密度可调：

- `f_min`：起始频率（$10^{-5} \sim 10^6$ Hz）
- `f_max`：截止频率（$10^{-5} \sim 10^6$ Hz）
- 点/dec：每数量级的数据点数（$5 \sim 200$）

频率轴按对数均匀分布：

$$f_{\text{scan}} = \log_{10}^{-1}\left( \log_{10} f_{\min},\, \log_{10} f_{\max},\, N_{\text{pts}} \right) \tag{34}$$

### 6.4 阻抗计算与可视化

点击「计算」后软件：

1. 从元件编辑器解析电路表达式
2. 遍历频率扫描数组，计算各频率下的阻抗 $Z(\omega)$
3. 绘制 4 个子图：

| 位置 | 图形 | 横轴 | 纵轴 |
|:----:|:-----|:-----|:-----|
| (1,1) | Nyquist 图 | $Z'$ | $-Z''$ |
| (1,2) | Bode 幅值图 | $f$ | $\lvert Z\rvert$ |
| (2,1) | 相位图 | $f$ | $-\arctan(Z''/Z')$ (°) |
| (2,2) | 双轴实虚部图 | $f$ | $Z'$（左轴）+ $-Z''$（右轴） |

---

## 7. 常见问题与故障排除

### 7.1 K-K 校验问题

| 问题 | 可能原因 | 解决方法 |
|:-----|:---------|:---------|
| $\chi^2$ 过大 | 数据含电感噪声或高频干扰 | 启用 `add_ind=True`，或增大频率筛选上限 |
| $\mu < 0.85$ | 数据不符合 K-K 关系 | 检查数据质量，增大 `max_M`，检查测量中的非稳态效应 |
| 拟合耗时过长 | max_M 过大或数据点数过多 | 减小 max_M 至 30-50，或减少数据点数 |
| 仅 1 个 RC 元素 | 算法遭遇数值问题提前停止 | 增大 mu_cutoff 或改用优化模式 |

### 7.2 拟合问题（Fit）

| 问题 | 可能原因 | 解决方法 |
|:-----|:---------|:---------|
| 拟合不收敛 | 初始值不当或模型不匹配 | 检查元件表达式，调整参数初始值，解除固定参数约束 |
| 参数趋近边界 | 模型与数据不匹配 | 放宽参数上下界，或更换更合适的模型 |
| 残差过大或呈系统性 | 模型阶次不足 | 增加元件复杂度，或考虑使用 DRT 做无模型分析 |
| 报错奇异矩阵 | 参数之间存在强相关性 | 固定部分参数，减少自由参数数量 |

### 7.3 DRT / DOP / DDT 反演问题

| 问题 | 可能原因 | 解决方法 |
|:-----|:---------|:---------|
| $\gamma(\tau)$ 出现密集尖刺 | $\lambda$ 过小 | 增大 $\lambda$ (1e-3 ~ 1e-2) |
| $\gamma(\tau)$ 过于平滑无峰 | $\lambda$ 过大 | 减小 $\lambda$ (1e-5 ~ 1e-4) |
| 边界处 $\gamma$ 异常抬升 | 端点振荡效应 | 增加 $\tau$ 范围或改用 τ=1/f 策略 |
| gDRT 求解失败 | 电感区或电容区数据不足 | 检查频率范围是否覆盖零交点两侧 |
| cvxopt `domain error` | P 矩阵接近奇异 | 增大 $\lambda$ 或减少 $N_{\tau}$ |
| DCT 结果不稳定 | 小阻抗导致导纳爆炸 | 改用 DRT，或预处理数据 |
| QPHB 分布过尖/过噪 | `*_l2_λ0` 过小 | 增大对应块 λ0（如 DRT_l2_λ0、DOP_l2_λ0） |
| QPHB 分布过平 | `*_l2_λ0` 过大 | 减小对应块 λ0 |
| fit_ddt_p 与 2d 冲突 | 同时勾选 | 界面已互斥；只保留其一 |
| 切换方法后参数“丢了” | — | 切换不再重置用户设置；检查是否切到另一参数面板 |

### 7.4 通用问题

| 问题 | 可能原因 | 解决方法 |
|:-----|:---------|:---------|
| 导入数据后无显示 | 列格式不兼容 | 检查 CSV 是否包含表头，分隔符是否为逗号/制表符 |
| 图表无法放大 | — | 使用 matplotlib 工具栏的放大镜工具框选区域 |
| 导出 Excel 报错 | 缺少 openpyxl | 执行 `pip install openpyxl` |
| cvxopt 安装失败 | 编译环境问题 | 从 https://cvxopt.org/install/ 下载预编译 wheel |

### 7.5 参数推荐值速查

| 参数 | 用途 | 典型范围 | 默认值 |
|:-----|:-----|:---------|:------:|
| $\lambda$ (DRT/DCT) | Tikhonov 强度 | $10^{-5} \sim 10^{-2}$ | $10^{-3}$ |
| $\lambda$ (gDRT) | Tikhonov 强度 | $10^{-5} \sim 10^{-3}$ | $10^{-4}$ |
| $N_{\tau}$ | τ 离散点数 | $100 \sim 500$ | $400$ |
| 配点 / 权重 (legacy) | — | — | τ=1/f / 无加权 |
| `DRT_l2_λ0` | QPHB DRT 块 | $10\sim300$ | **140** |
| `DOP_l2_λ0` | QPHB DOP 块 | $1\sim50$ | **10** |
| `DDT_l2_λ0` | QPHB DDT 块 | $0.1\sim10$ | **1.0** |
| `rp_scale` | 数据缩放 | $10\sim100$ | $50$ |
| `max_iter` | QPHB 迭代 | $10\sim100$ | $30$ |
| $\mu_{\text{cutoff}}$ | K-K 阈值 | $0.80 \sim 0.95$ | $0.85$ |
| $M_{\max}$ | K-K 最大 RC | $30 \sim 100$ | $50$ |
| 点/dec | 仿真频率密度 | $5 \sim 200$ | $50$ |

### 7.6 方法选择指南

| 方法 | 适用场景 | 不适用场景 |
|:-----|:---------|:------------|
| **DRT** | 容性主导 EIS（电池、腐蚀等） | 强电感主导 |
| **DCT** | 扩散主导、高频低阻抗 | 导纳爆炸的极小阻抗噪声 |
| **gDRT** | 同时含电感区+电容区 | 单侧频带过窄 |
| **DRT-DOP / DOP** | CPE、分数阶、相位角连续分布 | 仅需离散 RC 峰时用 DRT 即可 |
| **DRT-DDT / DDT** | 有限/半无限/多孔扩散 | 无明显扩散特征 |
| **2D DDT** | 异常扩散 $p\neq 0.5$ | 点少、噪声大 |

---

## 8. 参考文献

1. Schönleber, M., Klotz, D., & Ivers-Tiffée, E. (2014). A method for improving the robustness of linear Kramers-Kronig validity tests. *Electrochimica Acta*, 131, 20-27. doi:10.1016/j.electacta.2014.02.057

2. Boukamp, B. A. (1995). A linear Kronig-Kramers transform test for immittance data validation. *Journal of the Electrochemical Society*, 142(6), 1885. doi:10.1149/1.2044210

3. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution. *Electrochimica Acta*, 184, 483-499. doi:10.1016/j.electacta.2015.09.097

4. Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. *Electrochimica Acta*, 167, 439-454. doi:10.1016/j.electacta.2015.03.123

5. Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics*, 2(2), 164-168.

6. Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.

7. Siroma, Z., Fujiwara, N., Yamazaki, S., Asahi, M., Nagai, T., & Ioroi, T. (2015). Mathematical solutions of comprehensive variations of a transmission-line model of the theoretical impedance of porous electrodes. *Electrochimica Acta*, 160, 313-322. doi:10.1016/j.electacta.2015.02.065

8. Plank, C., Rüther, T., Jahn, L., Schamel, M., Schmidt, J. P., Ciucci, F., & Danzer, M. A. (2024). A review on the distribution of relaxation times analysis: A powerful tool for process identification of electrochemical systems. *Journal of Power Sources*, 594, 233845. doi:10.1016/j.jpowsour.2023.233845

9. Py, B., Maradesa, A., & Ciucci, F. (2023). Theoretical and Experimental Study of the Distribution of Capacitive Times. *arXiv preprint*. GitHub: https://github.com/ciuccislab/EIS-DCT

10. Danzer, M. A. (2019). Generalized Distribution of Relaxation Times Analysis for the Characterization of Impedance Spectra. *Batteries*, 5, 53. doi:10.3390/batteries5030053

11. Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. *Electrochimica Acta*, 247, 1117-1129. doi:10.1016/j.electacta.2017.07.050

12. Maradesa, A., Py, B., Wan, T. H., Effat, M. B., & Ciucci, F. (2023). Selecting the regularization parameter in the distribution of relaxation times. *Journal of the Electrochemical Society*, 170, 030502. doi:10.1149/1945-7111/acbca4

13. Vogelsang, J., & Schiller, C. A. (2000). A new approach for the evaluation of impedance data by means of the Kramers-Kronig transformation. *Electrochimica Acta*, 45(22-23), 3837-3845.

---

## 附录：术语表

| 术语 | 英文 | 解释 |
|:-----|:-----|:-----|
| 电化学阻抗谱 | EIS | 测量电化学系统在不同频率正弦激励下的阻抗响应 |
| Nyquist 图 | Nyquist plot | 以 $Z'$ 为横轴、$-Z''$ 为纵轴的阻抗复平面图 |
| Bode 图 | Bode plot | 阻抗模 $\lvert Z\rvert$ 和相位角随频率变化的双对数/半对数图 |
| 传输线模型 | TLM | 描述多孔电极中离子/电子传输的分布式等效电路模型 |
| 弛豫时间分布 | DRT | 将阻抗谱转化为时间常数域的 $\gamma(\tau)$ 分布 |
| 容性时间分布 | DCT | 在导纳空间进行分布反演，基于 $Y=G_\infty+j\omega C_\infty+\int\frac{j\omega\tau}{1+j\omega\tau}\gamma(\tau)d\ln\tau$ 模型提取特征时间常数 |
| 广义 DRT | gDRT | 同时处理容性和感性响应的 DRT 方法 |
| 相位角分布 | DOP (Distribution of Phasances) | $\rho(\nu)$ 对 $(j\omega)^\nu$ 的分布，描述 CPE 等分数阶过程 |
| 扩散时间分布 | DDT (Distribution of Diffusion Times) | $q(\tau_D)$ 对有限/异常扩散核的分布；可扩展为 2D $q(\tau_D,p)$ |
| 分层贝叶斯正则化 | QPHB | hybrid-drt 中的迭代自适应正则化，按块使用 l2_λ0 |
| Tikhonov 正则化 | Tikhonov regularization | 通过惩罚解的粗糙度来抑制过拟合的数学方法 |
| 二次规划 | QP | 带线性约束的凸二次优化问题 |
| 复非线性最小二乘 | CNLS | 复数域的非线性曲线拟合方法 |
| Boukamp 加权 | Boukamp weighting | 权重 $w_n = 1/|Z_n|^2$ 的加权方案 |
| μ 准则 | mu criterion | K-K 校验中衡量拟合物理一致性的指标 |
| K-K 关系 | Kramers-Kronig relations | 因果系统实部和虚部之间的 Hilbert 变换关系 |
