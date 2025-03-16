# -*- coding:UTF-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.loader.bearing.XJTULoader import XJTULoader
from rulframework.data.loader.bearing.PHM2012Loader import PHM2012Loader
from rulframework.data.labeler.RulLabeler import RulLabeler
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.data.processor.KurtosisProcessor import KurtosisProcessor
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.metric.Evaluator import Evaluator
from rulframework.metric.end2end.MAE import MAE
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.util.Plotter import Plotter
from pytorch_tcn1 import TCN


class SharedTCN(nn.Module):
    def __init__(self, tcn_params):
        super(SharedTCN, self).__init__()
        self.tcn = TCN(
            num_inputs=tcn_params['num_inputs'],
            num_channels=tcn_params['num_channels'],
            kernel_size=tcn_params['kernel_size'],
            dropout=tcn_params['dropout'],
            causal=tcn_params['causal'],
            use_norm=tcn_params['use_norm'],
            activation=tcn_params['activation'],
            kernel_initializer=tcn_params['kernel_initializer'],
            use_skip_connections=tcn_params['use_skip_connections'],
            output_projection=None,
            use_gate=True
        )

    def forward(self, x):
        return self.tcn(x)


class FeatureBranch(nn.Module):
    def __init__(self, branch_type, input_size, hidden_size):
        super(FeatureBranch, self).__init__()
        self.branch_type = branch_type
        self.input_proj = nn.Linear(input_size, hidden_size)

        if branch_type == "LSTM":
            self.model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        elif branch_type == "Transformer":
            self.positional_encoding = PositionalEncoding(hidden_size, 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.input_proj(x)
        if self.branch_type == "LSTM":
            out, _ = self.model(x)
            return out[:, -1, :]
        elif self.branch_type == "Transformer":
            x = self.positional_encoding(x)
            out = self.model(x)
            return out[:, -1, :]


class ChannelAttention1D(nn.Module):
    """1D通道注意力机制"""

    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x形状: [B, C, L]
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))  # [B, C]
        max_out = self.mlp(self.max_pool(x).squeeze(-1))  # [B, C]
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(-1)  # [B, C, 1]
        return x * channel_weights


class SpatialAttention1D(nn.Module):
    """1D空间注意力机制"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x形状: [B, C, L]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        spatial_weights = self.sigmoid(self.conv(combined))  # [B, 1, L]
        return x * spatial_weights


class CBAM1D(nn.Module):
    """适用于时序数据的CBAM模块"""

    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention1D(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention1D(kernel_size)

    def forward(self, x):
        # x形状: [B, C, L]
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, tcn_params, hidden_size=32, proj_dim=64):
        super(FusionModel, self).__init__()
        self.tcn = SharedTCN(tcn_params)
        tcn_out_channels = tcn_params['num_channels'][-1]

        # 特征分支
        self.lstm_branch = nn.Sequential(
            FeatureBranch("LSTM", tcn_out_channels, hidden_size),
            nn.Linear(hidden_size, proj_dim),
            nn.GELU()
        )
        self.trans_branch = nn.Sequential(
            FeatureBranch("Transformer", tcn_out_channels, hidden_size),
            nn.Linear(hidden_size, proj_dim),
            nn.GELU()
        )

        # 特征融合模块
        self.cbam = CBAM1D(in_channels=2, reduction_ratio=1, kernel_size=3)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, 1)
        )

    def forward(self, x):
        # TCN特征提取
        base_feat = self.tcn(x).transpose(1, 2)  # [B, L, C]

        # 分支特征提取
        lstm_feat = self.lstm_branch(base_feat)  # [B, proj_dim]
        trans_feat = self.trans_branch(base_feat)  # [B, proj_dim]

        # 组合特征 (通道维度=2)
        combined = torch.stack([lstm_feat, trans_feat], dim=1)  # [B, 2, proj_dim]

        # 应用CBAM
        cbam_out = self.cbam(combined)  # [B, 2, proj_dim]

        # 特征融合（通道维度求和）
        fused = cbam_out.sum(dim=1)  # [B, proj_dim]

        return self.fc(fused)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 数据准备
    data_loader = XJTULoader(
        'C:/Users/Administrator/Desktop/zhiguo/数字孪生/剩余寿命预测/数据集/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets')
    # data_loader = PHM2012Loader('C:\\Users\\Administrator\\Desktop\\zhiguo\\数字孪生\\剩余寿命预测\\数据集\\PHM2012\\data')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    # feature_extractor = FeatureExtractor(KurtosisProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)
    bearing = data_loader("Bearing1_1", columns='Horizontal Vibration')
    # Plotter.raw(bearing)
    feature_extractor(bearing)
    stage_calculator(bearing)
    # Plotter.feature(bearing)

    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    train_set, test_set = data_set.split(0.7)
    val_set, test_set = test_set.split(0.33)

    # 设置随机种子
    set_random_seed(42)

    # 配置参数
    tcn_params = {
        'num_inputs': 1,
        'num_channels': [2, 8],
        'kernel_size': 128,
        'dropout': 0.1,
        'causal': True,
        'use_norm': 'weight_norm',
        'activation': 'relu',
        'kernel_initializer': 'xavier_uniform',
        'use_skip_connections': True
    }

    # 创建融合模型
    model = FusionModel(tcn_params, hidden_size=64, proj_dim=64)
    pytorch_model = PytorchModel(model)

    # 训练参数
    name = 'TCN_lstm_transformer_CBAM64_2_8_128'
    epochs = 150
    batch_size = 256
    lr = 0.001
    patience = 50 # 早停参数


    # 训练流程
    pytorch_model.train(train_set, val_set, test_set,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        model_name=name,
                        patience=patience)

    # 可视化与评估
    Plotter.loss(pytorch_model)
    result = pytorch_model.test(test_set, batch_size=batch_size)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name=name)

    evaluator = Evaluator()
    evaluator.add(MAE(), RMSE())
    evaluator(test_set, result, name=name)