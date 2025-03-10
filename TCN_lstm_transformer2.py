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
from pytorch_tcn1 import TCN

# nvidia-smi -l 0.2

class TCN_LSTM(nn.Module):
    def __init__(self, tcn_params, lstm_input_size, lstm_hidden_size, lstm_num_layers, output_size):
        super(TCN_LSTM, self).__init__()

        # TCN配置
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

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        features = self.get_features(x)
        return self.fc(features)

    def get_features(self, x):
        tcn_out = self.tcn(x)
        lstm_input = tcn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)
        return lstm_out[:, -1, :]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding
        return self.dropout(x)

class TransformerMain(nn.Module):
    def __init__(self, input_size, output_len, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerMain, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        transformer_out = self.transformer_encoder(x)
        return transformer_out[:, -1, :]


class TCN_Transformer(nn.Module):
    def __init__(self, tcn_params, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TCN_Transformer, self).__init__()
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
            output_projection=None
        )
        self.transformer = TransformerMain(
            input_size=input_size,
            output_len=1,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        features = self.get_features(x)
        return self.fc(features)

    def get_features(self, x):
        tcn_out = self.tcn(x)
        trans_input = tcn_out.transpose(1, 2)
        return self.transformer(trans_input)


class FusionModel(nn.Module):
    def __init__(self, tcn_lstm, tcn_trans, hidden_size):
        super(FusionModel, self).__init__()
        self.tcn_lstm = tcn_lstm
        self.tcn_trans = tcn_trans

        # 特征维度投影
        self.proj_lstm = nn.Linear(64, hidden_size)  # LSTM hidden_size
        self.proj_trans = nn.Linear(64, hidden_size)  # Transformer d_model

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 2),
            nn.Softmax(dim=1)
        )

        # 最终输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_feat = self.tcn_lstm.get_features(x)
        trans_feat = self.tcn_trans.get_features(x)

        # 特征投影
        lstm_proj = self.proj_lstm(lstm_feat)
        trans_proj = self.proj_trans(trans_feat)

        # 注意力计算
        combined = torch.cat([lstm_proj, trans_proj], dim=1)
        attn_weights = self.attention(combined)
        fused = attn_weights[:, 0:1] * lstm_proj + attn_weights[:, 1:2] * trans_proj

        return self.fc(fused)


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
    bearing = data_loader("Bearing1_1", 'Horizontal Vibration')
    feature_extractor(bearing)
    stage_calculator(bearing)
    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    train_set, test_set = data_set.split(0.7)
    val_set, test_set = test_set.split(0.33)

    # 设置随机种子
    set_random_seed(42)

    # 初始化组件
    tcn_lstm_params = {
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

    tcn_trans_params = {
        'num_inputs': 1,
        'num_channels': [2, 8],
        'kernel_size': 32,
        'dropout': 0.1,
        'causal': True,
        'use_norm': 'weight_norm',
        'activation': 'relu',
        'kernel_initializer': 'xavier_uniform',
        'use_skip_connections': True
    }

    # 创建基础模型
    lstm_model = TCN_LSTM(
        tcn_params=tcn_lstm_params,
        lstm_input_size=8,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        output_size=1
    )

    trans_model = TCN_Transformer(
        tcn_params=tcn_trans_params,
        input_size=8,
        d_model=64,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1
    )

    # 创建融合模型
    fusion_model = FusionModel(lstm_model, trans_model, hidden_size=64)
    pytorch_model = PytorchModel(fusion_model)

    # 训练参数
    epochs = 150
    batch_size = 256
    lr = 0.001
    name = 'TCN_lstm_transformer_2_8_32+128'

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