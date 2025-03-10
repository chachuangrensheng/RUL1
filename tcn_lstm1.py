# -*- coding:UTF-8 -*- #
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
from rulframework.metric.end2end.MSE import MSE
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.util.Plotter import Plotter
from pytorch_tcn1 import TCN

# nvidia-smi -l 0.2

class TCN_LSTM(nn.Module):
    def __init__(self, tcn_params, lstm_input_size, lstm_hidden_size, lstm_num_layers, output_size):
        super(TCN_LSTM, self).__init__()

        # TCN配置参数解包
        num_inputs = tcn_params['num_inputs']
        num_channels = tcn_params['num_channels']
        kernel_size = tcn_params['kernel_size']
        dropout = tcn_params['dropout']
        causal = tcn_params['causal']
        use_norm = tcn_params['use_norm']
        activation = tcn_params['activation']
        kernel_initializer = tcn_params['kernel_initializer']
        use_skip_connections = tcn_params['use_skip_connections']

        # 创建TCN特征提取器（移除最后的全连接层）
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            use_norm=use_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_skip_connections=use_skip_connections,
            use_gate=True, # 启用门控
            output_projection=None  # 移除TCN的输出投影层
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

        # tcn_features = self.tcn(x.unsqueeze(1))  # 增加通道维度 (batch, 1, seq_len)
        tcn_features = self.tcn(x)
        # 调整形状以适应LSTM输入 (batch, seq_len, features)
        lstm_input = tcn_features.transpose(1, 2)  # 转换为(batch, seq_len, features)

        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)

        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])

        return out

    # 设置随机种子的函数
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
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTULoader(
        'C:/Users/Administrator/Desktop/zhiguo/数字孪生/剩余寿命预测/数据集/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets')

    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader("Bearing1_1", 'Horizontal Vibration')
    # Plotter.raw(bearing)
    feature_extractor(bearing)
    stage_calculator(bearing)
    # Plotter.feature(bearing)

    # 生成训练数据
    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    train_set, test_set = data_set.split(0.7)
    # 使用 split 方法划分验证集和测试集
    val_set, test_set = test_set.split(0.33)  # 按 10% 划分验证集和测试集


    # train_set.clear()

    # # 通过其他轴承增加训练数据
    # for bearing_name in ['Bearing1_1', 'Bearing1_2',
    #                      'Bearing2_3', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5',
    #                      'Bearing3_3']:
    #     bearing_train = data_loader(bearing_name)
    #     feature_extractor(bearing_train)
    #     stage_calculator(bearing_train)
    #     another_dataset = generator(bearing_train)
    #     another_dataset, _ = another_dataset.split(0.7)
    #     train_set.append(another_dataset)
    #     # test_set.append(_)

    # train_set, test_set_1 = data_set.split(0.7)

    # 设置随机种子
    seed = 42
    set_random_seed(seed)

    # 参数设置
    epochs = 150
    batch_size = 256
    name = 'TCN1_LSTM_2_8_32'
    lr = 0.001

    # 定义组合模型
    model = TCN_LSTM(
        tcn_params={
            'num_inputs': 1,
            'num_channels': [2, 8],
            'kernel_size': 32,
            'dropout': 0.1,
            'causal': True,
            'use_norm': 'weight_norm',
            'activation': 'relu',
            'kernel_initializer': 'xavier_uniform',
            'use_skip_connections': True
        },
        lstm_input_size=8,  # 匹配TCN最后一层输出通道数
        lstm_hidden_size=64,
        lstm_num_layers=2,
        output_size=1
    )

    # 将模型包装为PytorchModel
    pytorch_model = PytorchModel(model)

    # 训练流程
    pytorch_model.train(train_set, val_set, test_set, epochs=epochs, batch_size=batch_size, lr=lr,
                        model_name=name)

    Plotter.loss(pytorch_model)

    # 做出预测并画预测结果
    result = pytorch_model.test(test_set, batch_size=batch_size)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name=name)

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), RMSE())
    evaluator(test_set, result, name=name)