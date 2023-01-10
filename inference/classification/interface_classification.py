from classification.interface_covid import Classification as ClassificationCovid
from classification.interface_severe import Classification as ClassificationSevere


class Interface:

    def __init__(self, path_model_covid, path_model_severe, path_model_backup, split_list=['0', '1', '2', '3', '4']):
        self.interface_covid = ClassificationCovid(split_list=split_list, path_model=path_model_covid)
        self.interface_severe = ClassificationSevere(split_list=split_list, path_model=path_model_severe, path_model_backup=path_model_backup)

    def predict(self, img_itk, lung_np):
        prob_covid = self.interface_covid.predict(img_itk, lung_np)
        prob_severe = self.interface_severe.predict(img_itk, lung_np)
        return prob_covid, prob_severe


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    import os
    import SimpleITK as sitk
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    path_csv = '/mnt/sda/work/competitions/STOIC2021/code/train_model/preparation/information.csv'
    path_root_img = '/mnt/sda/work/competitions/STOIC2021/data/mha'
    path_root_lung = '/mnt/sda/work/competitions/STOIC2021/data/lung_mask'

    val_split = 3
    classification = Interface(path_model_covid='/mnt/sda/work/competitions/STOIC2021/code/stoic2021-finalphase/inference/classification/weight_covid',
                               path_model_severe='/mnt/sda/work/competitions/STOIC2021/code/stoic2021-finalphase/inference/artifact/weight_severe',
                               path_model_backup='/mnt/sda/work/competitions/STOIC2021/code/stoic2021-finalphase/inference/classification/weight_severe',
                               split_list=[str(val_split)])

    df = pd.read_csv(path_csv)
    df_val = df.loc[df['split'] == val_split]
    # df_val = df[(df['split'] == val_split) & (df['probCOVID'] == 1)]
    case_list = df_val['case'].to_list()
    gt_covid_list = df_val['probCOVID'].to_list()
    gt_severe_list = df_val['probSevere'].to_list()
    pred_covid_list, pred_severe_list = [], []
    for case, gt_covid, gt_severe in tqdm(zip(case_list, gt_covid_list, gt_severe_list), total=len(case_list)):
        case = str(case)
        path_file = os.path.join(path_root_img, case + '.mha')
        img_itk = sitk.ReadImage(path_file)
        path_lung = os.path.join(path_root_lung, case + '_lung.nii.gz')
        lung_itk = sitk.ReadImage(path_lung)
        lung_np = sitk.GetArrayFromImage(lung_itk).astype(np.uint8)

        prob_covid, prob_severe = classification.predict(img_itk, lung_np)
        pred_covid_list.append(prob_covid)
        pred_severe_list.append(prob_severe)

        print('file:', case, 'covid:', gt_covid, prob_covid, 'severe:', gt_severe, prob_severe)

    auc_covid = roc_auc_score(gt_covid_list, pred_covid_list)
    auc_severe = roc_auc_score(gt_severe_list, pred_severe_list)

    gt_severe_list = [gt_severe_list[i] for i in range(len(gt_severe_list)) if gt_covid_list[i] == 1]
    pred_severe_list = [pred_severe_list[i] for i in range(len(pred_severe_list)) if gt_covid_list[i] == 1]
    auc_covid_severe = roc_auc_score(gt_severe_list, pred_severe_list)
    print('covid:', auc_covid, 'severe:', auc_severe, 'covid severe:', auc_covid_severe)