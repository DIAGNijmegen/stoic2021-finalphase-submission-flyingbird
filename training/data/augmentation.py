from monai.transforms import Rand3DElasticDict, RandGaussianNoise, RandGaussianSmooth, RandAdjustContrast, RandFlipDict, RandScaleIntensity
import numpy as np


def aug_instantiation(shape_train, device, allow_missing_keys=False):
    aug_spatial = Rand3DElasticDict(
        keys=['img', 'mask'],
        sigma_range=(10, 13),
        magnitude_range=(0, 256),
        spatial_size=shape_train,
        prob=1.,
        rotate_range=[np.pi / 2., np.pi / 6., np.pi / 6.],
        scale_range=[0.1, 0.1, 0.1],
        mode=['bilinear', 'nearest'],
        padding_mode=['border', 'zeros'],
        as_tensor_output=True,
        device=device,
        allow_missing_keys=allow_missing_keys
    )

    aug_gaussian_noise = RandGaussianNoise(
        mean=0.,
        std=0.1,
        prob=0.25
    )

    aug_gaussian_smooth = RandGaussianSmooth(
        sigma_x=(0.5, 1.5),
        sigma_y=(0.5, 1.5),
        sigma_z=(0.5, 1.5),
        prob=0.25
    )

    aug_intensity = RandScaleIntensity(
        factors=(-0.3, 0.3),
        prob=0.
    )

    aug_contrast = RandAdjustContrast(
        gamma=(0.65, 0.5),
        prob=0.
    )

    aug_flip_0 = RandFlipDict(
        keys=['img', 'mask'],
        prob=0.5,
        spatial_axis=0,
        allow_missing_keys=allow_missing_keys
    )

    aug_flip_1 = RandFlipDict(
        keys=['img', 'mask'],
        prob=0.5,
        spatial_axis=1,
        allow_missing_keys=allow_missing_keys
    )

    aug_flip_2 = RandFlipDict(
        keys=['img', 'mask'],
        prob=0.5,
        spatial_axis=2,
        allow_missing_keys=allow_missing_keys
    )

    return [(aug_spatial, 'dict'), (aug_gaussian_noise, 'img'), (aug_gaussian_smooth, 'img'), (aug_intensity, 'img'), (aug_contrast, 'img'), (aug_flip_0, 'dict'), (aug_flip_1, 'dict'), (aug_flip_2, 'dict')]
