# -*- coding:UTF-8 -*- #
import torch
from torch import nn
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
torch.cuda.empty_cache()
# nvidia-smi -l 0.2

class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 输入形状转换: (batch_size, 2048) -> (batch_size, seq_len=2048, input_size=1)
        x = x.unsqueeze(-1)  # 增加特征维度

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # 输出形状: (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out


def predict_in_batches(model, test_set, batch_size):
    """
    手动分批处理数据并进行推理。
    """
    model.eval()
    x = test_set.x

    outputs = []
    with torch.no_grad():  # 关闭梯度计算
        for i in range(0, len(x), batch_size):  # 按批次处理数据
            # 获取当前批次的数据，并转换为 PyTorch 张量
            batch_x = torch.tensor(x[i:i + batch_size], dtype=torch.float32).to(device)

            # 模型推理
            batch_output = model(batch_x)

            # 保存结果
            outputs.append(batch_output.cpu())  # 将输出移回 CPU 并保存

    # 拼接所有批次的输出
    outputs = torch.cat(outputs, dim=0)
    return outputs

# 以下部分保持不变，训练与预测流程不需要修改
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
    # train_set.clear()

    # 定义模型并训练
    model = PytorchModel(LSTM_Model(input_size=1))
    model.train(train_set, batch_size=32, epochs=100)
    Plotter.loss(model)

    # 做出预测并画预测结果
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    # result = model.test(test_loader)
    # result = model.test(test_set)
    # result = model.predict_in_batches(test_set, batch_size=16)
    # 调用分批测试函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 获取模型所在的设备 (CPU 或 GPU)

    outputs = predict_in_batches(LSTM_Model(input_size=1).to(device), test_set, batch_size=16)
    outputs = outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
    # 包装为 Result 对象
    class Result:
        def __init__(self, outputs):
            self.outputs = outputs  # 将模型输出保存为属性


    result = Result(outputs=outputs)
    Plotter.rul_end2end(test_set, result, is_scatter=False, name='LSTM')

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE())
    evaluator(test_set, result)


