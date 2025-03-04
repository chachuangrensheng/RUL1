# -*- coding:UTF-8 -*- #
"""
@filename:TCN.py
"""

from torch import nn
import torch
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
from pytorch_tcn import TCN

# nvidia-smi -l 0.2



if __name__ == '__main__':
    # use_cache = True

    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTULoader('D:\桌面\数字孪生\剩余寿命预测\数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader("Bearing1_1", 'Horizontal Vibration')
    Plotter.raw(bearing)
    feature_extractor(bearing)
    stage_calculator(bearing)
    Plotter.feature(bearing)

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


    # 设置随机种子
    seed = 42
    set_random_seed(seed)



    # 定义模型并训练
    tcn_model = TCN(
        num_inputs=1,
        num_channels=[8, 14],
        kernel_size=4,
        dilations=None,
        dilation_reset=None,
        dropout=0.1,
        causal=True,
        use_norm='weight_norm',
        activation='relu',
        kernel_initializer='xavier_uniform',
        use_skip_connections=True,
        input_shape='NCL',
        embedding_shapes=None,
        embedding_mode='add',
        use_gate=False,
        lookahead=0,
        output_projection=1,
        output_activation=None,
    )

    model = PytorchModel(tcn_model)
    model.train(train_set, val_set, test_set, 150, batch_size=64, lr=0.001, model_name='TCN')
    Plotter.loss(model)

    # 做出预测并画预测结果
    result = model.test(test_set)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name='TCN')

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE())
    evaluator(test_set, result)

    MAE: 0.0895
    MSE: 0.0161
    RMSE: 0.1271
