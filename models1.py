# -*- coding:UTF-8 -*- #

import numpy as np
import random
import torch
import torch.nn as nn
import torch.fft


# 融合了双分支卷积特征提取、正交约束、时频融合和注意力机制
class AttentionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, hidden_size=32):
        super().__init__()
        # 结合正交性特征提取<button class="citation-flag" data-index="5">和注意力机制<button class="citation-flag" data-index="2">
        self.conv_branch1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size)
        )
        self.conv_branch2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_size * 2, hidden_size * 2 // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2 // 8, hidden_size * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(hidden_size * 4, 64)

    def forward(self, x):
        # 正交分支处理<button class="citation-flag" data-index="5">
        out1 = self.conv_branch1(x)
        out2 = self.conv_branch2(x)
        combined = torch.cat([out1, out2], dim=1)

        # 通道注意力机制<button class="citation-flag" data-index="2">
        attn = self.attention(combined)
        features = combined * attn

        # 时频融合特征
        time_domain = torch.mean(features, dim=2)
        freq_domain = torch.mean(torch.abs(torch.fft.fft(features, dim=2)), dim=2)

        return self.fc(torch.cat([time_domain, freq_domain], dim=1))


class EnhancedLSTM(nn.Module):
    def __init__(self, lstm_input=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.feature_extractor = AttentionFeatureExtractor()
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 输入形状处理
        x = x.unsqueeze(1)  # (batch, 1, time_steps)
        features = self.feature_extractor(x)  # (batch, 64)
        lstm_in = features.unsqueeze(1)  # (batch, 1, 64)
        lstm_out, _ = self.lstm(lstm_in)
        return self.fc(lstm_out.squeeze(1))


# 多尺度卷积和统计网络方法
class StatisticalFeatureLayer(nn.Module):
    def __init__(self):
        super(StatisticalFeatureLayer, self).__init__()

    def forward(self, x):
        # 时域特征
        # 计算均值
        mean = torch.mean(x, dim=2)

        # 计算方差
        var = torch.var(x, dim=2)

        # 计算最大值
        max_val = torch.max(x, dim=2)[0]

        # 计算最小值
        min_val = torch.min(x, dim=2)[0]

        # 计算均方根（Root Mean Square）
        rms = torch.sqrt(torch.mean(x**2, dim=2))

        # 计算峰峰值（Peak-to-peak）
        peak_to_peak = max_val - min_val

        # 计算偏度（Skewness），衡量数据分布的不对称性
        residual = x - mean.unsqueeze(2)

        skewness = torch.mean(residual ** 3, dim=2) / (var ** 1.5 + 1e-8)

        # 峰度计算
        kurtosis = torch.mean(residual ** 4, dim=2) / (var ** 2 + 1e-8)

        # 频域特征
        fft_signal = torch.fft.fft(x, dim=2)
        magnitude = torch.abs(fft_signal[:, :, :x.shape[2] // 2])
        freq = torch.fft.fftfreq(x.shape[2])[:x.shape[2] // 2].to(x.device)

        # 平均频率
        avg_freq = torch.sum(magnitude * freq, dim=2) / (torch.sum(magnitude, dim=2) + 1e-8)

        # 频率方差
        freq_var = torch.sum(magnitude * (freq - avg_freq.unsqueeze(2)) ** 2, dim=2) / (
                    torch.sum(magnitude, dim=2) + 1e-8)

        # 频率标准差
        freq_std = torch.sqrt(freq_var)

        # 频率均方根
        freq_rms = torch.sqrt(torch.mean(magnitude ** 2, dim=2))

        # 特征拼接 (8时域 + 4频域 = 12个特征)
        return torch.cat([mean, var, max_val, min_val, rms, peak_to_peak,
                          skewness, kurtosis, avg_freq, freq_var, freq_std, freq_rms], dim=1)


class MultiScaleConvNet(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvNet, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, hidden_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        self.stat_layer = StatisticalFeatureLayer()

    def forward(self, x):
        # x: (batch, channels, time_steps)
        features = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            stat_features = self.stat_layer(conv_out)
            features.append(stat_features)

        # 拼接多尺度特征
        out = torch.cat(features, dim=1)
        return out


class MultiScaleStatLSTM(nn.Module):
    def __init__(self, conv_params, lstm_hidden_size, lstm_num_layers, output_size):
        super(MultiScaleStatLSTM, self).__init__()
        self.feature_extractor = MultiScaleConvNet(**conv_params)

        # 计算LSTM输入大小
        num_stats = 12  # 每个尺度的统计特征数量
        num_scales = len(conv_params['kernel_sizes'])
        lstm_input_size = conv_params['hidden_channels'] * num_scales * num_stats

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # x: (batch, time_steps) -> (batch, 1, time_steps)
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)  # (batch, feature_dim)
        lstm_in = features.unsqueeze(1)  # (batch, 1, feature_dim)
        lstm_out, _ = self.lstm(lstm_in)
        out = self.fc(lstm_out.squeeze(1))
        return out
