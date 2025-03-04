from abc import ABC, abstractmethod
import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import TensorDataset, DataLoader
from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result


class ABCModel(ABC):
    """
    预测器的内核
    对不同深度学习框架的适配器
    所有预测器必须聚合一个ABCPredictable
    模型必须实现ABCPredictable中的predict抽象方法
    使预测器与模型能够规范接口联合使用
    """

    @abstractmethod
    def __init__(self, model):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss(self) -> list:
        raise NotImplementedError

    # def test(self, test_set: Dataset) -> Result:
    #     return Result(outputs=self(test_set.x))

    def test(self, test_set: Dataset, batch_size: int = 8) -> "Result":
        """
        测试模型性能
        :param test_set: 测试数据集
        :param batch_size: 批量大小（默认为8）
        :return: Result对象，包含测试结果
        """
        # 将模型设置为评估模式
        self.model.eval()

        # 创建数据加载器，分批加载测试数据
        x_test = torch.tensor(test_set.x, dtype=self.dtype, device=self.device)
        y_test = torch.tensor(test_set.y, dtype=self.dtype, device=self.device)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

        all_outputs = []
        device = next(self.model.parameters()).device  # 获取模型所在的设备

        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in test_loader:
                # 将输入数据移动到模型所在的设备
                # 如果 inputs 是列表，则将其转换为张量
                if isinstance(inputs, list):
                    inputs = torch.tensor(inputs, dtype=torch.float32)

                # 将输入数据移动到模型所在的设备
                inputs = inputs.to(device)

                # 前向传播
                outputs = self.model(inputs)

                # 将输出移回CPU并转换为NumPy数组
                all_outputs.append(outputs.cpu().numpy())

                # 拼接所有批次的输出
            all_outputs = np.concatenate(all_outputs, axis=0)

            # 返回结果
            return Result(outputs=all_outputs)

    @abstractmethod
    def train(self, train_set: Dataset, epochs: int = 100,
              batch_size: int = 128, weight_decay: float = 0,
              criterion=None, optimizer=None):
        raise NotImplementedError
