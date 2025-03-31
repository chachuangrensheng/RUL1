# -*- coding:UTF-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.data.loader.bearing.XJTULoader import XJTULoader
from rulframework.data.loader.bearing.PHM2012Loader import PHM2012Loader
from rulframework.data.labeler.RulLabeler import RulLabeler
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.system.Logger import Logger
from rulframework.util.Plotter import Plotter

if __name__ == '__main__':
    # data_loader = XJTULoader('D:\桌面\数字孪生\剩余寿命预测\数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets')
    data_loader = XJTULoader(
        'C:/Users/Administrator/Desktop/zhiguo/数字孪生/剩余寿命预测/数据集/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets')

    # data_loader = PHM2012Loader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    feature_extractor = FeatureExtractor(RMSProcessor(2048))
    # feature_extractor = FeatureExtractor(KurtosisProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 2048)

    bearing = data_loader("Bearing1_1", 'Horizontal Vibration')
    Plotter.raw(bearing)
    feature_extractor(bearing)
    stage_calculator(bearing)
    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    Plotter.feature(bearing, y_data=data_set.y)
    Logger.info(str(bearing))