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
from rulframework.data.Dataset import Dataset
from torch.utils.data import ConcatDataset, random_split
from pytorch_tcn1 import TCN

# nvidia-smi -l 0.2

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
                num_layers=2,
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
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=3)

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
    """适用于特征融合的通道注意力"""

    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 共享参数的MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2):
        """
        输入:
            feat1 - [B, proj_dim]
            feat2 - [B, proj_dim]
        输出:
            加权融合后的特征 [B, proj_dim]
        """
        # 拼接特征形成通道维度
        combined = torch.stack([feat1, feat2], dim=1)  # [B, 2, proj_dim]

        # 通道注意力计算
        avg_out = self.mlp(self.avg_pool(combined).squeeze(-1))  # [B, 2]
        max_out = self.mlp(self.max_pool(combined).squeeze(-1))  # [B, 2]
        channel_weights = self.sigmoid(avg_out + max_out)  # [B, 2]

        # 加权融合
        weighted_feat = channel_weights[:, 0:1] * feat1 + channel_weights[:, 1:2] * feat2
        return weighted_feat


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

        # 通道注意力模块
        self.channel_attn = ChannelAttention1D(num_channels=2)  # 2个特征分支

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, 1)
        )

    def forward(self, x):
        # TCN特征提取
        base_feat = self.tcn(x).transpose(1, 2)  # [B, seq_len, channels]

        # 分支特征提取
        lstm_feat = self.lstm_branch(base_feat)  # [B, proj_dim]
        trans_feat = self.trans_branch(base_feat)  # [B, proj_dim]

        # 通道注意力融合
        fused = self.channel_attn(lstm_feat, trans_feat)  # [B, proj_dim]

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
    # 训练参数
    Name = 'TCN_lstm_transformer_LOO_'
    epochs = 2
    batch_size = 128
    lr = 0.001
    patience = 30  # 早停参数
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

    # 数据准备
    data_loader = XJTULoader(
        'D:\桌面\数字孪生\剩余寿命预测\数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 轴承列表
    bearings = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5']

    # 预处理所有轴承数据并生成数据集
    all_datasets = {}
    for name in bearings:
        bearing = data_loader(name, columns='Horizontal Vibration')
        feature_extractor(bearing)
        stage_calculator(bearing)
        generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
        dataset = generator(bearing)
        all_datasets[name] = dataset

    # 留一法交叉验证
    results = []
    for i in range(len(bearings)):
        test_name = bearings[i]
        train_names = [name for j, name in enumerate(bearings) if j != i]

        # 使用自定义Dataset的append方法合并训练集
        combined_train = Dataset()  # 创建空数据集
        for name in train_names:
            combined_train.append(all_datasets[name])  # 追加每个训练轴承的数据
        # train_datasets = [all_datasets[name] for name in train_names]
        # combined_train = ConcatDataset(train_datasets)

        # 划分训练集和验证集 (80%训练, 20%验证)
        # train_size = int(0.8 * len(combined_train))
        # val_size = len(combined_train) - train_size
        # train_subset, val_subset = random_split(combined_train, [train_size, val_size])
        train_subset, val_subset = combined_train.split(0.8)
        test_set = all_datasets[test_name]

        # 初始化新模型
        model = FusionModel(tcn_params, hidden_size=64, proj_dim=128)
        pytorch_model = PytorchModel(model)

        # 训练模型
        pytorch_model.train(train_subset, val_subset, None,  # 测试集独立
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            model_name=Name+test_name,
                            patience=patience)

        # 绘制训练损失
        Plotter.loss(pytorch_model)

        # 测试评估
        test_result = pytorch_model.test(test_set, batch_size=batch_size)
        evaluator = Evaluator()
        evaluator.add(MAE(), RMSE())
        # 调用评估器并获取返回的评估结果字典
        evaluation = evaluator(test_set, test_result, name=Name)
        # 显式转换结果为浮点型
        evaluation = {k: float(v) for k, v in evaluation.items()}  # 新增类型转换
        results.append(evaluation)

        # 绘制测试结果
        Plotter.rul_end2end(test_set, test_result, is_scatter=False, name=Name)
        del evaluator, evaluation

    # 计算平均指标
    mae_values = [res['MAE'] for res in results]
    rmse_values = [res['RMSE'] for res in results]
    print(f'LOO Cross-Validation Results:')
    print(f'Average MAE: {np.mean(mae_values):.4f}')
    print(f'Average RMSE: {np.mean(rmse_values):.4f}')


