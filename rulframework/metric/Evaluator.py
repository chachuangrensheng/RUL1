from typing import List
import os  # 新增导入

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric import ABCMetric
from rulframework.system.Logger import Logger


class Evaluator:
    """
    指标评价器
    先使用add_metric添加需要的指标
    再调用evaluate计算所有的指标
    """

    def __init__(self) -> None:
        self.metrics: List[ABCMetric] = []

    def __str__(self) -> str:
        return super().__str__()

    def add(self, *args: ABCMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)

    def __call__(self, test_set: Dataset, result: Result, name=None) -> {}:
        """
        根据已经添加的评价指标开始计算
        :param result:
        :param test_set:
        :return:
        """
        # 验证输入的合法性
        sample_num = test_set.x.shape[0]
        if sample_num != result.outputs.shape[0]:
            raise Exception(f'测试样本量：{sample_num}与测试结果数量：{result.outputs.shape[0]} 不匹配')

        evaluation = {}
        string = f'Performance evaluation of {name + test_set.name}:'
        for metric in self.metrics:
            e = metric(test_set, result)
            evaluation[metric.name] = e
            string += f'\n\t{metric.name}: {e}'

        # 新增文件保存逻辑
        save_dir = 'txt'
        os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
        file_path = os.path.join(save_dir, f'{name + test_set.name}.txt')

        with open(file_path, 'w') as f:
            f.write(string)

        Logger.info('\n' + string)
        return evaluation