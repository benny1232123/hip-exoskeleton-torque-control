# 外骨骼算法 - 生物力矩控制器复现

基于 Pan 2023 和 Kang 2021 论文，使用 Camargo 2021 数据集复现生物力矩控制器（Biological Torque Controller）。

## 项目概述

本项目从 Camargo 2021 开放下肢生物力学数据集中提取健康人平地行走数据，计算归一化髋关节力矩，并实现基于高斯求和的生物力矩控制器参考曲线生成。

## 文件结构

```
└── gait_data/                         # 数据处理目录
    ├── .venv_mat2csv/                 # Python 虚拟环境（用于读取 MATLAB 文件）
    ├── Part1/ Part2/ Part3/           # 解压数据（AB06-AB30，22人）
    │
    ├── process_part1.py               # 处理 Part 1 数据
    ├── process_part2_3_batch.py       # 批量处理 Part 2/3 数据
    ├── pan2023_full_dataset.py        # 主复现脚本（22人完整数据集）
    │
    └── processed/                     # 处理结果
        ├── levelground_all.csv        # 合并全部数据（22人，280万行）
        │
        ├── gait_curves_all_subjects.png           # 图1：健康步态曲线
        ├── biological_torque_controller_all.png   # 图2：控制器参数
        └── biological_torque_controller_demo_all.png  # 图3：单试验演示
```

## 数据来源

**Camargo et al. 2021** - A comprehensive, open-source dataset of lower limb biomechanics

- Part 1: https://data.mendeley.com/datasets/fcgm3chfff/1
- Part 2: https://data.mendeley.com/datasets/k9kvm5tn3f/1
- Part 3: https://data.mendeley.com/datasets/jj3r5f9pnf/1

**数据集详情：**
- 22 名健康受试者（AB06-AB30）
- 仅使用**平地行走（level ground walking）**数据
- 采样率: IK/ID 200Hz

## 主要结果

### 图1：健康人步态生物力学曲线

![gait_curves_all_subjects](gait_data/processed/gait_curves_all_subjects.png)

**22个健康受试者平地行走的平均曲线（680个步态周期）**

| 子图 | 内容 | 说明 |
|------|------|------|
| 左上 | 髋关节屈曲/伸展角度 | 蓝=右腿，红=左腿，单位：度 |
| 右上 | 膝关节屈曲/伸展角度 | 单位：度 |
| 左下 | 髋关节力矩 | 正=屈曲，负=伸展，单位：Nm/kg |
| 右下 | 踝关节背屈/跖屈角度 | 单位：度 |

**关键发现：**
- 髋关节力矩峰值屈曲：**3.79 Nm/kg @ 62%** 步态周期
- 髋关节力矩峰值伸展：**-7.46 Nm/kg @ 0%** 步态周期

---

### 图2：生物力矩控制器参数

![biological_torque_controller_all](gait_data/processed/biological_torque_controller_all.png)

**生物力矩控制器的设计参数和参考曲线**

| 子图 | 内容 | 说明 |
|------|------|------|
| 左上 | 4个辅助等级的力矩曲线 | Level 0(0Nm)、1(~2Nm)、2(~4Nm)、3(~6Nm) |
| 右上 | 辅助时序图 | 蓝=支撑相(0-60%)，绿=摆动相(60-100%) |
| 左下 | 归一化髋力矩 vs 控制器参考曲线 | 黑线=真实力矩，红虚线=高斯拟合 |
| 右下 | 22人各自的髋力矩曲线 | 彩色=个人，黑粗线=平均 |

**高斯拟合参数：**
- 屈曲辅助：位置μ=60%，宽度σ=8%，幅度A可调（0~0.08 Nm/kg）
- 伸展辅助：位置μ=10%，宽度σ=8%，幅度A可调

---

### 图3：单试验控制器演示

![biological_torque_controller_demo_all](gait_data/processed/biological_torque_controller_demo_all.png)

**在 AB06 受试者的一个试验上演示控制器工作过程**

| 子图 | 内容 | 说明 |
|------|------|------|
| 第1行 | 髋关节角度 | 蓝线=滤波后，红色三角=步态周期峰值检测 |
| 第2行 | 估计的步态相位 | 锯齿波，0-100%循环 |
| 第3行 | 辅助力矩 | 根据当前相位实时计算的髋关节辅助力矩 |
| 第4行 | 叠加图 | 蓝=髋角度，红=辅助力矩 |

**关键结果：**
- 步态相位估计：基于髋关节角度峰值检测，每个峰值=步态周期90%
- 辅助力矩：在屈曲相(45-75%)施加正力矩，在伸展相(90-20%)施加负力矩

---

## 算法原理

### 核心思路

```
离线阶段：Camargo数据集 → 归一化力矩 → 高斯拟合 → 可调参考曲线
在线阶段：髋关节传感器 → 峰值检测 → 步态相位 → 查表输出力矩
```

### 高斯求和公式

```python
τ(t) = A₁ × exp(-0.5 × ((t - μ₁) / σ₁)²)   # 屈曲辅助（正力矩）
     + A₂ × exp(-0.5 × ((t - μ₂) / σ₂)²)   # 伸展辅助（负力矩）
```

### 辅助等级

| 等级 | 幅度A | 辅助力矩（75kg） | 用途 |
|------|-------|------------------|------|
| Level 0 | 0 | 0 Nm | 无辅助（对照） |
| Level 1 | 0.03 | ~2 Nm | 轻度辅助 |
| Level 2 | 0.05 | ~4 Nm | 中度辅助 |
| Level 3 | 0.08 | ~6 Nm | 重度辅助 |

## 使用方法

### 1. 环境配置

```bash
cd gait_data
python -m venv .venv_mat2csv
.venv_mat2csv\Scripts\activate
pip install numpy>=2.2.2 mat-io scipy pandas matplotlib
```

### 2. 处理数据

```bash
# 处理 Part 1
.venv_mat2csv\Scripts\python.exe process_part1.py

# 处理 Part 2/3
.venv_mat2csv\Scripts\python.exe process_part2_3_batch.py
```

### 3. 运行复现

```bash
.venv_mat2csv\Scripts\python.exe pan2023_full_dataset.py
```

## 参考文献

1. Camargo, J. et al. (2021). A comprehensive, open-source dataset of lower limb biomechanics. *Data in Brief*, 36, 107036.
2. Pan, Y. et al. (2023). Effects of Bilateral Assistance for Hemiparetic Gait Post-Stroke Using a Powered Hip Exoskeleton.
3. Kang, I. et al. (2021). Real-Time Gait Phase Estimation for Robotic Hip Exoskeleton Control During Multimodal Locomotion.
