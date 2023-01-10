from typing import Dict
from pathlib import Path
import SimpleITK
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm
from lungmask import mask as LungMask
from classification.interface_classification import Interface as InterfaceClassification


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
                ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/"),
        )
        
        self.lung_segmentor = LungMask.get_model('unet', 'R231', modelpath='./segmentation/weight/unet_r231-d5d2fc3d.pth')

        self.classifier = InterfaceClassification(path_model_covid='./classification/weight_covid', path_model_severe='./artifact/weight_severe', path_model_backup='./classification/weight_severe')

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        lung_mask_zyx = LungMask.apply(input_image, batch_size=32, model=self.lung_segmentor)
        prob_covid, prob_severe = self.classifier.predict(input_image, lung_mask_zyx)

        return {
            COVID_OUTPUT_NAME: float(prob_covid),
            SEVERE_OUTPUT_NAME: float(prob_severe)
        }


import pandas as pd
import os
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class TestWQ:

    def __init__(self):
        self.predictor = StoicAlgorithm()

    def predict(self, path_csv, path_root_img):
        df = pd.read_csv(path_csv)

        # check
        df_val = df.loc[(df['split'] == 4)]

        file_list = df_val['file'].to_list()
        gt_covid_list = df_val['probCOVID'].to_list()
        gt_severe_list = df_val['probSevere'].to_list()
        pred_covid_list, pred_severe_list = [], []
        for file in tqdm(file_list):
            # if file != '5647.mha':
            if file != '23.mha':
                continue
            path_file = os.path.join(path_root_img, file)
            img_itk = sitk.ReadImage(path_file)
            ans_dict = self.predictor.predict(input_image=img_itk)
            prob_covid, prob_severe = ans_dict[COVID_OUTPUT_NAME], ans_dict[SEVERE_OUTPUT_NAME]
            print(f'{file} covid:{prob_covid} severe:{prob_severe}')
            pred_covid_list.append(prob_covid)
            pred_severe_list.append(prob_severe)

        auc_covid = roc_auc_score(gt_covid_list, pred_covid_list)
        auc_severe = roc_auc_score(gt_severe_list, pred_severe_list)

        gt_severe_list = [gt_severe_list[i] for i in range(len(gt_severe_list)) if gt_covid_list[i] == 1]
        pred_severe_list = [pred_severe_list[i] for i in range(len(pred_severe_list)) if gt_covid_list[i] == 1]
        auc_covid_severe = roc_auc_score(gt_severe_list, pred_severe_list)
        print('covid:', auc_covid, 'severe:', auc_severe, 'covid severe:', auc_covid_severe)


if __name__ == "__main__":
    StoicAlgorithm().process()

    # t = TestWQ()
    # t.predict(path_csv='/mnt/sda/work/competitions/STOIC2021/metadata/information_v4.csv',
    #           path_root_img='/mnt/sda/work/competitions/STOIC2021/data/mha')