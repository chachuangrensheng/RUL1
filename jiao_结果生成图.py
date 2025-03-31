# -*- coding:UTF-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from rulframework.util.Plotter import Plotter


name = 'bearing1_4_jiao_TCN_lstm_transformer_CA'

# 读取保存的 CSV 文件
loaded_data = np.loadtxt(name + 'results.csv', delimiter=',', skiprows=1)
combined_true_loaded = loaded_data[:, 0]  # 第一列是真实值
combined_pred_loaded = loaded_data[:, 1]  # 第二列是预测值

# 计算每个 fold 的起始索引
n_samples = len(combined_pred_loaded)
n_folds = 5
base_size, remainder = divmod(n_samples, n_folds)

fold_boundaries = [0]  # 初始化起始位置
current = 0
for i in range(n_folds):
    # 根据余数调整前几个 fold 的长度
    fold_size = base_size + 1 if i < remainder else base_size
    current += fold_size
    fold_boundaries.append(current)

# 生成对应的 x 轴坐标（保持原始顺序）
x_axis_loaded = np.arange(n_samples)

# 使用加载的数据绘制合并后的结果图
Plotter.rul_end2end_combined(
    true_rul=combined_true_loaded,
    pred_rul=combined_pred_loaded,
    x_axis=x_axis_loaded,
    fold_boundaries=fold_boundaries,
    fold_names=[f"Fold {i + 1}" for i in range(n_folds)],
    is_scatter=False,
    name=name
)