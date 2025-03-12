import torch
from numpy import ndarray
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.system.Logger import Logger


class PytorchModel(ABCModel):
    """
    剩余寿命预测模型
    对pytorch神经网络的封装
    """

    @property
    def loss(self) -> list:
        return self.train_losses

    def __init__(self, model: nn.Module, device=None, dtype=None) -> None:
        """
        :param model:pytorch模型
        :param device: 设备（cpu或cuda）
        :param dtype: 参数类型
        """
        # 初始化设备
        if device is None:
            # self.device = torch.device('cpu')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 初始化模型参数类型
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        # 初始化模型
        self.model = model.to(device=self.device, dtype=self.dtype)

        # 用于保存每次epoch的训练损失
        self.train_losses = []
        Logger.info(f'\n<< Successfully initialized model:\n\tdevice: {self.device}\n\tdtype: {self.dtype}')

    def __call__(self, x: ndarray) -> ndarray:
        input_data = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.model(input_data)
        return output.cpu().numpy()

    def train(self, train_set: Dataset, val_set: Dataset = None, test_set: Dataset = None, epochs=100,
              batch_size=128, weight_decay=0, lr=0.001,
              criterion=None, optimizer=None, model_name=None,
              patience=30, min_delta=0):
        """
        训练模型
        :param train_set: 训练数据集
        :param val_set: 验证数据集（可选）
        :param lr: 学习率
        :param optimizer: 优化器（默认：Adam，学习率0.001）
        :param weight_decay: 正则化系数
        :param batch_size: 批量大小
        :param epochs: 最大迭代次数
        :param criterion: 损失函数
        :param model_name: 模型名称（用于保存）
        :param patience: 早停法容忍轮数
        :param min_delta: 早停法最小变化阈值
        :return: 无返回值
        """
        save_dir = "pth"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_name + "best_model.pth")

        Logger.info('Start training model...')

        if criterion is None:
            criterion = nn.MSELoss()

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        x_train = torch.tensor(train_set.x, dtype=self.dtype, device=self.device)
        y_train = torch.tensor(train_set.y, dtype=self.dtype, device=self.device)
        if isinstance(criterion, nn.CrossEntropyLoss):
            y_train = y_train.squeeze().to(dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)

        early_stop_counter = 0
        best_val_loss = float('inf')

        if val_set is not None:
            x_val = torch.tensor(val_set.x, dtype=self.dtype, device=self.device)
            y_val = torch.tensor(val_set.y, dtype=self.dtype, device=self.device)
            if isinstance(criterion, nn.CrossEntropyLoss):
                y_val = y_val.squeeze().to(dtype=torch.long)
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

        if test_set is not None:
            x_test = torch.tensor(test_set.x, dtype=self.dtype, device=self.device)
            y_test = torch.tensor(test_set.y, dtype=self.dtype, device=self.device)
            if isinstance(criterion, nn.CrossEntropyLoss):
                y_test = y_test.squeeze().to(dtype=torch.long)
            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            Logger.debug(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.10f}")

            if val_set is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                Logger.debug(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.10f}")

                # 早停法逻辑
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), save_path)
                    early_stop_counter = 0
                    Logger.info(f"Best model saved with validation loss: {best_val_loss:.10f}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        Logger.info(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                        break  # 提前终止训练

        # 加载最佳模型
        if val_set is not None:
            self.model.load_state_dict(torch.load(save_path))
            Logger.info("Loaded the best model based on validation loss.")

        Logger.info('Model training completed!!!')

        # 加载最佳模型
        self.model.load_state_dict(torch.load(save_path))

        # 对训练集进行预测
        Logger.info('Start predicting on training set...')
        self.model.eval()
        train_predictions = []
        train_labels = []
        train_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                train_loss_sum += loss.item()
                train_predictions.append(outputs.cpu().numpy())
                train_labels.append(labels.cpu().numpy())
        train_predictions = np.vstack(train_predictions)
        train_labels = np.vstack(train_labels)
        Logger.info(f"Training set loss: {train_loss_sum / len(train_loader):.10f}")

        # 保存训练集预测结果到 .xlsx 文件
        train_df = pd.DataFrame({
            'Train_Predictions': train_predictions.flatten(),
            'Train_Labels': train_labels.flatten()
        })
        train_df.to_excel(model_name + test_set.name + 'train_predictions.xlsx', index=False)
        Logger.info('Training set predictions saved to train_predictions.xlsx')

        # 对测试集进行预测
        if test_set is not None:
            Logger.info('Start predicting on test set...')
            test_predictions = []
            test_labels = []
            test_loss_sum = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss_sum += loss.item()
                    test_predictions.append(outputs.cpu().numpy())
                    test_labels.append(labels.cpu().numpy())
            test_predictions = np.vstack(test_predictions)
            test_labels = np.vstack(test_labels)
            Logger.info(f"Test set loss: {test_loss_sum / len(test_loader):.10f}")

            # 保存测试集预测结果到 .xlsx 文件
            test_df = pd.DataFrame({
                'Test_Predictions': test_predictions.flatten(),
                'Test_Labels': test_labels.flatten()
            })
            test_df.to_excel(model_name + test_set.name + 'test_predictions.xlsx', index=False)
            Logger.info('Test set predictions saved to test_predictions.xlsx')
