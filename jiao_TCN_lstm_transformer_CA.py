# -*- coding:UTF-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
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
from rulframework.system.Logger import Logger
from rulframework.data.Dataset import Dataset
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
    # 数据准备
    # data_loader = XJTULoader(
    #     'D:\桌面\数字孪生\剩余寿命预测\数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets')
    data_loader = XJTULoader(
        'C:/Users/Administrator/Desktop/zhiguo/数字孪生/剩余寿命预测/数据集/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets')

    # data_loader = PHM2012Loader('C:\\Users\\Administrator\\Desktop\\zhiguo\\数字孪生\\剩余寿命预测\\数据集\\PHM2012\\data')

    feature_extractor = FeatureExtractor(RMSProcessor(2048))
    # feature_extractor = FeatureExtractor(KurtosisProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 2048)
    bearing = data_loader("Bearing1_4", columns='Horizontal Vibration')
    Plotter.raw(bearing)
    feature_extractor(bearing)
    stage_calculator(bearing)


    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    Plotter.feature(bearing, y_data=data_set.y)
    # 初始化K折交叉验证 (K=5)
    kf = KFold(n_splits=5, shuffle=False)
    fold_results = []

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
    model = FusionModel(tcn_params, hidden_size=64, proj_dim=128)


    # 训练参数
    name = 'bearing1_1_2_jiao_TCN_lstm_transformer_CA'
    epochs = 150
    batch_size = 256
    lr = 0.001
    patience = 50  # 早停参数

    # 在交叉验证循环外初始化存储所有预测结果的容器
    all_true = []
    all_pred = []
    fold_boundaries = []  # 记录每个fold验证集的起始位置

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_set.x)):
        Logger.info(f"========== Processing Fold {fold_idx + 1}/5 ==========")

        # 分割当前fold的数据
        x_train = data_set.x[train_idx]
        y_train = data_set.y[train_idx]
        z_train = data_set.z[train_idx]  # 新增z的分割
        x_val = data_set.x[val_idx]
        y_val = data_set.y[val_idx]
        z_val = data_set.z[val_idx]  # 新增z的分割

        # 创建当前fold的训练集和验证集
        train_fold = Dataset(x_train, y_train, z_train)  # 同时传入x,y,z
        val_fold = Dataset(x_val, y_val, z_val, name=f"Fold{fold_idx}_Val" )

        # 初始化新的模型（确保每个fold独立）
        pytorch_model = PytorchModel(model)  # 使用原始模型参数

        # 训练当前fold的模型
        pytorch_model.train(
            train_set=train_fold,
            val_set=val_fold,  # 验证集来自交叉验证划分
            test_set=None,  # 不传入测试集
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_name=f"{name}_fold{fold_idx}",
            patience=patience
        )

        # 加载当前fold的最佳模型
        best_model_path = os.path.join("pth", f"{name}_fold{fold_idx}best_model.pth")
        pytorch_model.model.load_state_dict(torch.load(best_model_path))

        # 在验证集上评估当前模型性能
        result = pytorch_model.test(val_fold, batch_size=batch_size)


        # 记录当前fold的验证结果
        all_true.append(val_fold.y.squeeze())  # 假设y是二维数组，压缩为1维
        all_pred.append( result.outputs.squeeze())  # 强制转为1维数组

        # 记录当前fold验证集的起始索引（用于绘图分界线）
        fold_start = len(np.concatenate(all_true[:-1])) if fold_idx > 0 else 0  # 计算累积长度
        fold_boundaries.append(fold_start)

        # 记录评估指标
        evaluator = Evaluator()
        evaluator.add(MAE(), RMSE())
        metrics = evaluator(val_fold, result, name=f"{name}_fold{fold_idx}")
        fold_results.append(metrics)

        # 可选：保存每个fold的训练过程可视化
        Plotter.loss(pytorch_model)
        Plotter.rul_end2end(val_fold, result, is_scatter=False, name=f"{name}_fold{fold_idx}")


    # 计算交叉验证平均指标
    final_mae = sum(float(fold['MAE']) for fold in fold_results) / len(fold_results)
    final_rmse = sum(float(fold['RMSE']) for fold in fold_results) / len(fold_results)

    # 保存评估指标于txt文件中
    with open(f"{name}_cross_validation_results.txt", "w") as file:
        file.write(f"Average MAE: {final_mae:.4f}\n")
        file.write(f"Average RMSE: {final_rmse:.4f}\n")


    Logger.info(f"\nK-Fold Cross Validation Final Results (MAE/RMSE):")
    Logger.info(f"Average MAE: {final_mae:.4f}")
    Logger.info(f"Average RMSE: {final_rmse:.4f}")

    # 添加最后一个分界点（数据总长度）
    fold_boundaries.append(len(np.concatenate(all_true)))

    # 合并所有结果
    combined_true = np.concatenate(all_true)
    combined_pred = np.concatenate(all_pred)

    # 将两个数组合并为两列
    combined = np.column_stack((combined_true, combined_pred))

    # 保存为CSV文件，设置表头和分隔符，并移除注释符号
    np.savetxt(name + 'results.csv', combined, delimiter=',', header='true,pred', comments='')


    # 生成对应的x轴坐标（保持原始顺序）
    x_axis = np.arange(len(combined_true))

    # 绘制合并后的结果图
    Plotter.rul_end2end_combined(
        true_rul=combined_true,
        pred_rul=combined_pred,
        x_axis=x_axis,
        fold_boundaries=fold_boundaries,
        fold_names=[f"Fold {i + 1}" for i in range(5)],
        is_scatter=False,
        name=name
    )




    # # 训练流程
    # pytorch_model.train(train_set, val_set, test_set,
    #                     epochs=epochs,
    #                     batch_size=batch_size,
    #                     lr=lr,
    #                     model_name=name,
    #                     patience=patience)
    #
    # # 可视化与评估
    # Plotter.loss(pytorch_model)
    # result = pytorch_model.test(test_set, batch_size=batch_size)
    # Plotter.rul_end2end(test_set, result, is_scatter=False, name=name)
    #
    # evaluator = Evaluator()
    # evaluator.add(MAE(), RMSE())
    # evaluator(test_set, result, name=name)