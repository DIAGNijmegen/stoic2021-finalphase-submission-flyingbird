import pandas as pd
import numpy as np
from lungmask import mask as LungMask
import os
import SimpleITK as sitk
from tqdm import tqdm
import random
import multiprocessing
from skimage import morphology
from scipy.ndimage import zoom


def generate_case_split_dict(path_csv):
    # df = pd.read_excel(path_csv, sheet_name='Sheet1')
    df = pd.read_csv(path_csv)
    case_list, split_list = df['case'].to_list(), df['split'].to_list()
    case_split_dict = {case: split for case, split in zip(case_list, split_list)}
    np.save('./case_split_mapping.npy', case_split_dict)


def generate_lung_mask(path_root_img, path_csv, path_root_save):
    if not os.path.exists(path_root_save):
        os.makedirs(path_root_save)

    df = pd.read_csv(path_csv)
    df = df[df['probCOVID'] == 1]
    case_list = df['PatientID'].to_list()

    lung_segmentor = LungMask.get_model('unet', 'R231', modelpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet_r231-d5d2fc3d.pth'))

    for case in tqdm(case_list):
        path_mha = os.path.join(path_root_img, str(case) + '.mha')
        img_itk = sitk.ReadImage(path_mha)
        lung_mask_zyx = LungMask.apply(img_itk, batch_size=32, model=lung_segmentor)
        lung_mask_zyx = lung_mask_zyx.astype(np.uint8)
        lung_mask_zyx_itk = sitk.GetImageFromArray(lung_mask_zyx)
        sitk.WriteImage(lung_mask_zyx_itk, os.path.join(path_root_save, str(case) + '.nii.gz'))


def generate_train(path_root_img, path_root_lung, path_csv, path_mapping_dict, path_root_save, path_csv_save):
    if not os.path.exists(path_root_save):
        os.makedirs(path_root_save)

    case_split_dict = np.load(path_mapping_dict, allow_pickle=True).item()

    df = pd.read_csv(path_csv)
    df = df[df['probCOVID'] == 1]

    case_list = df['PatientID'].to_list()
    probCOVID_list = df['probCOVID'].to_list()
    probSevere_list = df['probSevere'].to_list()

    split_list = []
    for case in case_list:
        if case in case_split_dict:
            split_list.append(case_split_dict[case])
        else:
            split_list.append(random.randint(0, 4))

    age_list = []
    pool = multiprocessing.Pool(8)
    for case in case_list:
        path_img = os.path.join(path_root_img, str(case) + '.mha')
        path_lung = os.path.join(path_root_lung, str(case) + '.nii.gz')
        age_list.append(pool.apply_async(_process, args=(path_img, path_lung, path_root_save, case)))
    pool.close()
    pool.join()
    age_list = [age.get() for age in age_list]

    df = pd.DataFrame({'case': case_list, 'probCOVID': probCOVID_list, 'probSevere': probSevere_list, 'age': age_list, 'split': split_list})
    df.to_csv(path_csv_save, index=False)


def _process(path_img, path_lung, path_root_save, case):
    img_itk, img_np, lung_np = _load_data(path_img, path_lung)
    lung_np = _process_mask(lung_np)
    img_zoom_np_list, lung_zoom_np_list = _crop_and_zoom(img_itk, img_np, lung_np)
    _save_np(img_zoom_np_list, path_root_save, case)
    age = _get_side_info(img_itk, case)
    return age


def _load_data(path_img, path_lung):
    img_itk = sitk.ReadImage(path_img)
    img_np = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    lung_itk = sitk.ReadImage(path_lung)
    lung_np = sitk.GetArrayFromImage(lung_itk).astype(np.uint8)
    return img_itk, img_np, lung_np


def _process_mask(lung_np):
    lung_np[lung_np > 0] = 1
    lung_np = morphology.remove_small_objects(lung_np.astype(bool), min_size=100).astype(np.uint8)
    return lung_np


def _crop_and_zoom(img_itk, img_np, lung_np):
    assert img_np.shape == lung_np.shape, 'shape not the same'

    pos_z, pos_y, pos_x = np.where(lung_np > 0)
    z1_ori, y1_ori, x1_ori = np.min(pos_z), np.min(pos_y), np.min(pos_x)
    z2_ori, y2_ori, x2_ori = np.max(pos_z), np.max(pos_y), np.max(pos_x)
    spacing_x, spacing_y, spacing_z = img_itk.GetSpacing()

    shift_z_max, shift_y_max, shift_x_max = int(np.around(9. / spacing_z)), int(np.around(9. / spacing_y)), int(np.around(9. / spacing_x))
    shift_z_min, shift_y_min, shift_x_min = int(np.around(3. / spacing_z)), int(np.around(3. / spacing_y)), int(np.around(3. / spacing_x))
    img_zoom_np_list, lung_zoom_np_list, p_zoom_np_list = [], [], []
    for i in range(3):
        if i != 2:
            shift_z, shift_y, shift_x = random.randint(shift_z_min, shift_z_max), random.randint(shift_y_min, shift_y_max), random.randint(shift_x_min, shift_x_max)
        else:
            shift_z, shift_y, shift_x = int(np.around(6. / spacing_z)), int(np.around(6. / spacing_y)), int(np.around(6. / spacing_x))
        z, y, x = lung_np.shape
        z1, y1, x1 = max(0, z1_ori - shift_z), max(0, y1_ori - shift_y), max(0, x1_ori - shift_x)
        z2, y2, x2 = min(z - 1, z2_ori + shift_z), min(y - 1, y2_ori + shift_y), min(x - 1, x2_ori + shift_x)

        img_crop_np = img_np[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]
        lung_crop_np = lung_np[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]

        zoom_factor = np.array([256, 256, 256]) / np.array(img_crop_np.shape)
        img_zoom_np = zoom(img_crop_np.astype(np.float32), zoom_factor, order=1)
        lung_zoom_np = zoom(lung_crop_np.astype(np.uint8), zoom_factor, order=0)

        img_zoom_np_list.append(img_zoom_np)
        lung_zoom_np_list.append(lung_zoom_np)

    return img_zoom_np_list, lung_zoom_np_list


def _get_side_info(img_itk, case):
    # age and sex
    age_mapping = {'35': 0, '45': 1, '55': 2, '65': 3, '75': 4, '85': 5}
    try:
        age = img_itk.GetMetaData('PatientAge')
        if len(age) > 2:
            age = age[1:-1]
        age = age_mapping[age]
    except:
        age = 5

    return age


def _save_np(img_np_list, path_root_save, case):
    for i in range(len(img_np_list)):
        if i != len(img_np_list) - 1:
            np.savez(os.path.join(path_root_save, str(case) + '_' + str(i) + '.npz'), img=img_np_list[i])
        else:
            np.savez(os.path.join(path_root_save, str(case) + '_6mm.npz'), img=img_np_list[i])


def prepare(path_root_img, path_root_lung, path_csv, path_mapping_dict, path_root_npz_save, path_csv_save):
    generate_lung_mask(path_root_img=path_root_img,
                       path_csv=path_csv,
                       path_root_save=path_root_lung)

    generate_train(path_root_img=path_root_img,
                   path_root_lung=path_root_lung,
                   path_csv=path_csv,
                   path_mapping_dict=path_mapping_dict,
                   path_root_save=path_root_npz_save,
                   path_csv_save=path_csv_save)


if __name__ == '__main__':
    # generate_case_split_dict(path_csv='/mnt/sda/work/competitions/STOIC2021/code/train_model/preparation/information.csv')

    case_split_dict = np.load('./case_split_mapping.npy', allow_pickle=True).item()
    print(len(case_split_dict))
    print(type(case_split_dict))
    # for key, val in case_split_dict.items():
    #     print(type(key), key)
    #     print(type(val), val)