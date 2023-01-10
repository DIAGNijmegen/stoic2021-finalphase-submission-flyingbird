import torch.backends.cudnn
from data.dataset_severe import DatasetP
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import os
from models.resnet import resnet
import time
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
from torch.multiprocessing import set_start_method
from torch.cuda.amp import autocast, GradScaler
from preparation.generate_data import prepare
from configs.config import config


class Train:
    def __init__(self,
                 learning_rate=3e-4,
                 weight_decay=1e-4,
                 batch_size=1,
                 epoch=200,
                 clip=(-1024, 1024),
                 mean=None,
                 std=None,
                 shape_train=(255, 255, 255),
                 path_data_train='/mnt/sdb/nodule/npz',
                 path_csv='',
                 path_save='/mnt/sda/work/projects/nodule/CenterNet/experiments/trial_1',
                 path_pretrain='',
                 num_classes=5,
                 val_frequency=5,
                 val_split=0):

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epoch = epoch
        self.clip = clip
        self.mean = mean
        self.std = std
        self.shape_train = shape_train
        self.path_data_train = path_data_train
        self.path_csv = path_csv
        self.path_save = path_save
        self.path_pretrain = path_pretrain
        self.num_classes = num_classes
        self.val_frequency = val_frequency
        self.val_split = val_split

        self.device = torch.device('cuda:0')
        self.device_data_process = torch.device('cuda:1')

        # prepare dataset
        self.dataloader_train, self.dataset_val = self._prepare_dataset()
        # prepare model
        self.model = self._prepare_model()
        # prepare optimizer
        self.optimizer, self.lr_scheduler = self._prepare_optimizer()
        # prepare criterion
        self.criterion_ce = self._prepare_criterion()

        self.scaler = GradScaler()

    def _prepare_dataset(self):
        dataset_train = DatasetP(path_root=self.path_data_train,
                                 path_csv=self.path_csv,
                                 val_split=self.val_split,
                                 clip=self.clip,
                                 mean=self.mean,
                                 std=self.std,
                                 shape_train=self.shape_train,
                                 device=self.device_data_process,
                                 path_save=self.path_save)

        dataloader_train = DataLoader(dataset_train,
                                      self.batch_size,
                                      num_workers=6,
                                      pin_memory=False,
                                      shuffle=True,
                                      drop_last=True)

        val_list = self._prepare_val()

        print('prepare dataset done!', 'num data:', len(dataset_train))
        return dataloader_train, val_list

    def _prepare_val(self):
        df = pd.read_csv(self.path_csv)
        df_val = df[(df['split'] == self.val_split) & (df['probCOVID'] == 1)]

        case_list = df_val['case'].to_list()
        path_list = [os.path.join(self.path_data_train, str(case) + '_6mm.npz') for case in case_list]

        data_age = df_val[['age']].to_numpy(copy=True)
        data_age = data_age / 5.
        info = data_age.astype(np.float32)

        severe_list = df_val['probSevere'].to_list()

        val_list = []
        for path, info_curr, severe in zip(path_list, info, severe_list):
            val_list.append({'path': path, 'info': info_curr, 'severe': severe})

        return val_list

    def _prepare_model(self):
        model = resnet(in_channels=1, out_channels=2, layers=[2, 2, 2, 2])

        if os.path.exists(self.path_pretrain):
            state_dict_pretrain = torch.load(self.path_pretrain, map_location='cpu')['state_dict']
            state_dict_pretrain_processed = {}
            for key in state_dict_pretrain:
                if 'module' in key:
                    state_dict_pretrain_processed[key.replace('module.', '')] = state_dict_pretrain[key]
                else:
                    state_dict_pretrain_processed[key] = state_dict_pretrain[key]

            state_dict = model.state_dict()
            for key in state_dict:
                if key in state_dict_pretrain_processed and state_dict[key].shape == state_dict_pretrain_processed[key].shape:
                    state_dict[key] = state_dict_pretrain_processed[key]
                    print('load pretrain model by key iteration, key:', key)
            model.load_state_dict(state_dict)

        model.to(self.device)
        return model

    def _prepare_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, self.learning_rate, epochs=self.epoch, steps_per_epoch=len(self.dataloader_train))
        print('prepare optimizer done')
        return optimizer, scheduler

    def _prepare_criterion(self):
        criterion_ce = torch.nn.CrossEntropyLoss()
        return criterion_ce

    def train(self):
        torch.backends.cudnn.benchmark = True

        auc_severe_best = 0
        auc_severe = self._val_one_epoch(0)
        auc_severe_best = self._save_model(auc_severe, auc_severe_best)
        for ep in range(1, self.epoch + 1):
            print('train ep:', ep)
            self._train_one_epoch(ep)
            if ep % self.val_frequency == 0:
                auc_severe = self._val_one_epoch(ep)
                auc_severe_best = self._save_model(auc_severe, auc_severe_best)

    def _save_model(self, auc_severe, auc_severe_best):
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        if auc_severe > auc_severe_best:
            torch.save({'state_dict': state_dict, 'auc_severe': auc_severe}, os.path.join(self.path_save, 'severe_best.pkl'))
            print('saveing model', 'auc_severe from', auc_severe_best, 'to', auc_severe)
            auc_severe_best = auc_severe

        return auc_severe_best

    def _train_one_epoch(self, ep):
        self.model.train()
        for it, (img, info, label_severe) in enumerate(self.dataloader_train, 0):
            img, info, label_severe = img.to(self.device), info.to(self.device), label_severe.to(self.device)

            with autocast():
                out, feature = self.model(img, info)
                loss = self.criterion_ce(out, label_severe)

            # lr_current = self.optimizer.state_dict()['param_groups'][0]['lr']

            # print('Train\tEpoch:', ep,
            #       '\tIter:', it,
            #       '\tLoss:', round(loss.item(), 5),
            #       '\tLR:', lr_current)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

    def _val_one_epoch(self, ep):
        self.model.eval()

        pred_severe_list, gt_severe_list = [], []
        for val_dict in tqdm(self.dataset_val):
            img_curr, info, label_severe = self._preprocess(val_dict)
            img_list = self._generate_img_list(img_curr, tta=True)
            pred = self._infer(img_list, info)
            pred_severe_list.append(pred)
            gt_severe_list.append(label_severe)

        auc_severe = self._evaluate(pred_severe_list, gt_severe_list)

        return auc_severe

    def _evaluate(self, pred_severe_list, gt_severe_list):
        auc_severe = roc_auc_score(gt_severe_list, pred_severe_list)
        return auc_severe

    def _infer(self, img_list, info):
        info = torch.from_numpy(info)
        info = info.to(self.device)

        pred_list = []
        for img in img_list:
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img)
            img = img.to(self.device)
            with torch.no_grad():
                out, _ = self.model(img, info)
                pred_list.append(torch.softmax(out, dim=1).cpu().numpy().squeeze()[1])
        pred = np.mean(pred_list)

        return pred

    def _preprocess(self, val_dict):
        # 'path': path, 'enc_info': enc_info, 'sc_info': sc_info, 'covid': covid, 'severe': severe
        path = val_dict['path']
        data = np.load(path, allow_pickle=True)
        img = data['img'].astype(np.float32)
        img[img < self.clip[0]] = self.clip[0]
        img[img > self.clip[1]] = self.clip[1]
        img = (img - self.mean) / self.std

        img = img[np.newaxis, ...]

        info = np.array([val_dict['info']], dtype=np.float32)

        label_severe = int(val_dict['severe'])

        return img, info, label_severe

    def _generate_img_list(self, img, tta=True):
        img_list = [img.copy()]
        if tta:
            img_list.append(np.flip(img, axis=1).copy())
            img_list.append(np.flip(img, axis=2).copy())
            img_list.append(np.flip(img, axis=3).copy())
        return img_list


def do_learning(data_dir, artifact_dir):
    """
    You can implement your own solution to the STOIC2021 challenge by editing this function.
    :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
        structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
    :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
        weights) that the inference Docker container needs. It is recommended to continuously update the contents of
        this directory during training.
    :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
        in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
        returned. Note: To limit the size of your inference Docker container, please make sure to only place files that
        are necessary for inference into artifact_dir.
    """

    set_start_method('spawn')

    prepare(path_root_img='/input/data/mha',
            path_root_lung='/scratch/lung',
            path_csv='/input/metadata/reference.csv',
            path_mapping_dict='/opt/train/preparation/case_split_mapping.npy',
            path_root_npz_save='/scratch/data_npz',
            path_csv_save='/scratch/information.csv')

    for split in [0, 1, 2, 3, 4]:
        config['path_save'] = os.path.join('/output/weight_severe', 'split' + str(split))
        config['val_split'] = split
        config['epoch'] = 6
        config['learning_rate'] = 3e-4
        config['weight_decay'] = 1e-4
        config['shape_train'] = (256, 256, 256)
        config['batch_size'] = 30
        config['val_frequency'] = 1
        config['path_data_train'] = '/scratch/data_npz'
        config['path_pretrain'] = os.path.join('/opt/train/pretrain/weight_severe', 'split' + str(split), 'severe_best.pkl')
        config['path_csv'] = '/scratch/information.csv'

        if not os.path.exists(config['path_save']):
            os.makedirs(config['path_save'])

        for key in config:
            print(key, config[key])

        train_net = Train(**config)
        train_net.train()

    return []
