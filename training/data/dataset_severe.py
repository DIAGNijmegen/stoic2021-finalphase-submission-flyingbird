import torch.utils.data
import numpy as np
import os
from data.augmentation import aug_instantiation
import pandas as pd


class DatasetP(torch.utils.data.Dataset):
    def __init__(self, path_root, path_csv, val_split, clip, mean, std, shape_train, device, path_save):
        super(DatasetP, self).__init__()

        self.path_root = path_root
        self.path_csv = path_csv
        self.val_split = val_split
        self.clip = clip
        self.mean = mean
        self.std = std
        self.path_save = path_save

        self.path_list, self.info_list, self.label_list = self._prepare_data_list()
        self.device = device
        self.aug_list = aug_instantiation(shape_train, self.device, True)

    def __getitem__(self, i):
        img = np.load(self.path_list[i], allow_pickle=True)['img'].astype(np.float32)
        img = self._process(img)
        info = np.array([self.info_list[i]], dtype=np.float32)
        severe = self.label_list[i]
        return img, info, severe

    def __len__(self):
        return len(self.path_list)

    def _prepare_data_list(self):
        df = pd.read_csv(self.path_csv)
        df_train = df[(df['split'] != self.val_split) & (df['probCOVID'] == 1)]

        case_list = df_train['case'].to_list()
        label_list = df_train['probSevere'].to_list()
        info_list = df_train['age'].to_list()

        path_list = []
        info_process_list = []
        label_process_list = []
        for case, label, info in zip(case_list, label_list, info_list):
            for suffix in ['0', '1', '6mm']:
                path = os.path.join(self.path_root, str(case) + '_' + suffix + '.npz')
                info = info / 5.
                path_list.append(path)
                info_process_list.append(info)
                label_process_list.append(label)

        return path_list, info_process_list, label_process_list

    def _augment(self, img):
        data_dict = {'img': img}
        for aug in self.aug_list:
            if aug[-1] == 'img':
                data_dict['img'] = aug[0](data_dict['img'])
            elif aug[-1] == 'dict':
                data_dict = aug[0](data_dict)

        return data_dict['img']

    def _process(self, img):
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)
        img = img.to(self.device)

        img[img < self.clip[0]] = self.clip[0]
        img[img > self.clip[1]] = self.clip[1]
        img = (img - self.mean) / self.std

        img = self._augment(img)

        img = img.cpu()

        return img
