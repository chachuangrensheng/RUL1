# -*- coding:UTF-8 -*-
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import random
from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.loader.bearing.XJTULoader import XJTULoader
from rulframework.data.labeler.RulLabeler import RulLabeler
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.metric.Evaluator import Evaluator
from rulframework.metric.end2end.MAE import MAE
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.util.Plotter import Plotter
from pytorch_tcn_sa import TCN


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
            use_sparse_attention=tcn_params['use_sparse_attention'],
            sparse_attention_heads=tcn_params['sparse_attention_heads'],
            sparse_window=tcn_params['sparse_window'],
            output_projection=None,
            use_gate=True
        )

    def forward(self, x):
        return self.tcn(x)


class FeatureBranch(nn.Module):
    def __init__(self, branch_type, input_size, hidden_size):
        super(FeatureBranch, self).__init__()
        self.branch_type = branch_type
        self.input_proj = nn.Linear(input_size, hidden_size)  # 新增维度适配层

        if branch_type == "LSTM":
            self.model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,  # 减少层数保持参数规模
                batch_first=True
            )
        elif branch_type == "Transformer":
            self.positional_encoding = PositionalEncoding(hidden_size, 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,  # 确保hidden_size能被nhead整除
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.input_proj(x)  # 维度转换
        if self.branch_type == "LSTM":
            out, _ = self.model(x)
            return out[:, -1, :]
        elif self.branch_type == "Transformer":
            x = self.positional_encoding(x)
            out = self.model(x)
            return out[:, -1, :]


class FusionModel(nn.Module):
    def __init__(self, tcn_params, hidden_size=32):  # 调整hidden_size为32
        super(FusionModel, self).__init__()
        # 共享TCN
        self.tcn = SharedTCN(tcn_params)

        # 获取TCN最终输出通道数
        tcn_out_channels = tcn_params['num_channels'][-1]

        # 两个特征分支
        self.lstm_branch = FeatureBranch("LSTM",
                                         input_size=tcn_out_channels,
                                         hidden_size=hidden_size)

        self.trans_branch = FeatureBranch("Transformer",
                                          input_size=tcn_out_channels,
                                          hidden_size=hidden_size)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1)
        )

        # 最终输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # TCN特征提取
        tcn_out = self.tcn(x)  # (batch, 8, seq)
        tcn_features = tcn_out.transpose(1, 2)  # (batch, seq, 8)

        # 分支处理
        lstm_feat = self.lstm_branch(tcn_features)  # 投影到hidden_size
        trans_feat = self.trans_branch(tcn_features)

        # 注意力融合
        combined = torch.cat([lstm_feat, trans_feat], dim=1)
        attn_weights = self.attention(combined)
        fused = attn_weights[:, 0:1] * lstm_feat + attn_weights[:, 1:2] * trans_feat

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

    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)
    bearing = data_loader("Bearing1_5", 'Horizontal Vibration')
    feature_extractor(bearing)
    stage_calculator(bearing)
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
        'use_skip_connections': True,
        'use_sparse_attention': True,  # 启用稀疏注意力
        'sparse_attention_heads': 4,
        'sparse_window': 16
    }

    # 创建融合模型
    model = FusionModel(tcn_params, hidden_size=64)
    pytorch_model = PytorchModel(model)

    # 训练参数
    epochs = 150
    batch_size = 16
    lr = 0.001
    name = 'TCN_sa_lstm_transformer_2_8_128'

    # 训练流程
    pytorch_model.train(train_set, val_set, test_set,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        model_name=name)

    # 可视化与评估
    Plotter.loss(pytorch_model)
    result = pytorch_model.test(test_set, batch_size=batch_size)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name=name)

    evaluator = Evaluator()
    evaluator.add(MAE(), RMSE())
    evaluator(test_set, result, name=name)