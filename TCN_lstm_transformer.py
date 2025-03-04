# -*- coding:UTF-8 -*-
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
from rulframework.metric.end2end.MSE import MSE
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.util.Plotter import Plotter
from pytorch_tcn1 import TCN

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TransformerMain(nn.Module):
    def __init__(self, input_size, output_len, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, batch_size,
                 device):
        super(TransformerMain, self).__init__()
        self.input_size = input_size
        self.output_len = output_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device

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
        last_out = transformer_out[:, -1, :]
        output = self.fc(last_out)
        return output


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
class AttentionFusion(nn.Module):
    def __init__(self, lstm_dim, transformer_dim):
        super().__init__()
        self.query = nn.Linear(lstm_dim, lstm_dim)
        self.key = nn.Linear(transformer_dim, transformer_dim)
        self.value = nn.Linear(transformer_dim, transformer_dim)
        self.scale = torch.sqrt(torch.FloatTensor([transformer_dim])).to(device)

    def forward(self, lstm_features, transformer_features):
        Q = self.query(lstm_features)
        K = self.key(transformer_features)
        V = self.value(transformer_features)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        fused_features = torch.matmul(attention_weights, V)
        return fused_features.squeeze(1)


class TCN_LSTM_Transformer(nn.Module):
    def __init__(self, tcn_params, lstm_hidden_size, lstm_num_layers,
                 d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        # TCN特征提取器
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

        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=tcn_params['num_channels'][-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Transformer分支
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 注意力融合层
        self.attention_fusion = AttentionFusion(lstm_hidden_size, d_model)

        # 输出层
        self.fc = nn.Linear(lstm_hidden_size + d_model, 1)

    def forward(self, x):
        # TCN特征提取
        tcn_features = self.tcn(x)  # (batch, channels, seq_len)

        # LSTM处理
        lstm_input = tcn_features.transpose(1, 2)  # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # Transformer处理
        transformer_input = tcn_features.transpose(1, 2)  # (batch, seq_len, channels)
        transformer_input = self.positional_encoding(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_features = transformer_out[:, -1, :]  # (batch, d_model)

        # 注意力融合
        fused_features = self.attention_fusion(lstm_features.unsqueeze(1),
                                               transformer_features.unsqueeze(1))

        # 组合特征并预测
        combined = torch.cat([lstm_features, fused_features], dim=1)
        output = self.fc(combined)
        return output


# 其他代码保持不变，替换模型实例化部分：
if __name__ == '__main__':
    data_loader = XJTULoader(
        'D:\桌面\数字孪生\剩余寿命预测\数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets')
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

    seed = 42
    set_random_seed(seed)

    epochs = 150
    batch_size = 64
    lr = 0.001
    name = 'TCN_LSTM_Transformer'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCN_LSTM_Transformer(
        tcn_params={
            'num_inputs': 1,
            'num_channels': [8, 14],  # 确保最后一层与LSTM和Transformer输入匹配
            'kernel_size': 4,
            'dropout': 0.1,
            'causal': True,
            'use_norm': 'weight_norm',
            'activation': 'relu',
            'kernel_initializer': 'xavier_uniform',
            'use_skip_connections': True
        },
        lstm_hidden_size=32,
        lstm_num_layers=2,
        d_model=32,  # 需与TCN最后一层通道数一致
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)

    pytorch_model = PytorchModel(model)
    pytorch_model.train(train_set, val_set, test_set, epochs=epochs, batch_size=batch_size, lr=lr,
                        model_name=name)

    Plotter.loss(pytorch_model)
    result = pytorch_model.test(test_set, batch_size=batch_size)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name=name)

    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE())
    evaluator(test_set, result, name=name)