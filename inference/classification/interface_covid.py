import numpy as np
from skimage import morphology
import SimpleITK as sitk
import os
from scipy.ndimage import zoom
import torch
from classification.resnet import resnet as resnet_covid


class Classification:

    def __init__(self, path_model, clip=(-1024.0, 512.0), mean=-236.88525, std=404.0286, shape_train=(256, 256, 256), split_list=['0', '1', '2', '3', '4']):
        self.path_model = path_model
        self.split_list = split_list
        self.model_dict = self._load_model()
        self.clip = clip
        self.mean = mean
        self.std = std
        self.shape_train = shape_train

    def predict(self, img_itk, lung_np):
        img, info = self._preprocess(img_itk, lung_np)
        img_list = self._generate_img_list(img, tta=True)
        prob_covid = self._infer(img_list, info)
        return prob_covid

    def _infer(self, img_list, info):
        covid_list = []
        for split in self.split_list:
            prob_covid = self._infer_one_split(img_list, info, split)
            covid_list.append(prob_covid)
        prob_covid = np.mean(covid_list)
        return prob_covid

    def _infer_one_split(self, img_list, info, split):
        model_covid = self.model_dict['covid_' + split]

        info = info[np.newaxis, ...]
        info = torch.from_numpy(info)
        info = info.to('cuda:0')

        covid_prob_list = []
        for img in img_list:
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img)
            img = img.to('cuda:0')
            with torch.no_grad():
                out_covid = model_covid(img, info)
                prob_covid = torch.softmax(out_covid, dim=1).cpu().numpy().squeeze()[1]
            covid_prob_list.append(prob_covid)

        prob_covid = np.mean(covid_prob_list)
        return prob_covid

    def _generate_img_list(self, img, tta=True):
        img_list = [img.copy()]
        if tta:
            img_list.append(np.flip(img, axis=1).copy())
            img_list.append(np.flip(img, axis=2).copy())
            img_list.append(np.flip(img, axis=3).copy())
        return img_list

    def _load_model(self):
        model_dict = {}
        for split in self.split_list:
            path_model_covid = os.path.join(self.path_model, 'split' + split, 'covid_best' + '.pkl')

            model_covid = resnet_covid(in_channels=1, out_channels=2, layers=[2, 2, 2, 2])
            covid_state_dict = torch.load(path_model_covid, map_location='cpu')['state_dict']
            model_covid.load_state_dict(covid_state_dict, strict=True)
            model_covid.eval()
            model_covid.to('cuda:0')

            model_dict['covid_' + split] = model_covid

        return model_dict

    def _preprocess(self, img_itk, lung_np):
        img_itk, img_np, lung_np = self._load_data(img_itk, lung_np)
        lung_np = self._process_mask(lung_np)
        img_zoom_np, lung_zoom_np = self._crop_and_zoom(img_itk, img_np, lung_np)

        info_dict = self._get_side_info(img_itk, img_np, lung_np)
        img, info = self._prepare_model_input(img_zoom_np, info_dict)
        return img, info

    def _load_data(self, img_itk, lung_np):
        img_np = sitk.GetArrayFromImage(img_itk).astype(np.float32)
        lung_np = lung_np.astype(np.uint8)
        return img_itk, img_np, lung_np

    def _process_mask(self, lung_np):
        lung_np[lung_np > 0] = 1
        lung_np = morphology.remove_small_objects(lung_np.astype(bool), min_size=100).astype(np.uint8)
        return lung_np

    def _crop_and_zoom(self, img_itk, img_np, lung_np):
        assert img_np.shape == lung_np.shape, 'shape not the same'

        pos_z, pos_y, pos_x = np.where(lung_np > 0)
        try:
            z1, y1, x1 = np.min(pos_z), np.min(pos_y), np.min(pos_x)
            z2, y2, x2 = np.max(pos_z), np.max(pos_y), np.max(pos_x)
        except:
            z1 = y1 = x1 = 0
            z2, y2, x2 = lung_np.shape[0] - 1, lung_np.shape[1] - 1, lung_np.shape[2] - 1
        spacing_x, spacing_y, spacing_z = img_itk.GetSpacing()

        shift_z, shift_y, shift_x = int(np.around(6. / spacing_z)), int(np.around(6. / spacing_y)), int(np.around(6. / spacing_x))
        z, y, x = lung_np.shape
        z1, y1, x1 = max(0, z1 - shift_z), max(0, y1 - shift_y), max(0, x1 - shift_x)
        z2, y2, x2 = min(z - 1, z2 + shift_z), min(y - 1, y2 + shift_y), min(x - 1, x2 + shift_x)

        img_crop_np = img_np[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]
        lung_crop_np = lung_np[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]

        zoom_factor = np.array(self.shape_train) / np.array(img_crop_np.shape)
        img_zoom_np = zoom(img_crop_np.astype(np.float32), zoom_factor, order=1)
        lung_zoom_np = zoom(lung_crop_np.astype(np.uint8), zoom_factor, order=0)

        return img_zoom_np, lung_zoom_np

    def _prepare_model_input(self, img, info_dict):
        img = img.astype(np.float32)

        img[img < self.clip[0]] = self.clip[0]
        img[img > self.clip[1]] = self.clip[1]
        img = (img - self.mean) / self.std

        img = img[np.newaxis, ...]

        info = np.array([info_dict['age']], dtype=np.float32)

        return img, info

    def _get_side_info(self, img_itk, img_np, mask_lung):
        # age and sex
        age_mapping = {'35': 0, '45': 0.2, '55': 0.4, '65': 0.6, '75': 0.8, '85': 1.}

        try:
            age = img_itk.GetMetaData('PatientAge')
            if len(age) > 2:
                age = age[1:-1]
            age = age_mapping[age]
        except:
            print('can not get patient age')
            age = 1.

        return {'age': age}
