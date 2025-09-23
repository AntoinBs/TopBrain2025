import numpy as np

import SimpleITK as sitk

from monai.transforms import (
    MapTransform,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd
)
from monai.data import MetaTensor

from torch.utils.data import Sampler

class CustomScaleIntensityRanged(MapTransform):
    """Custom intensity scaling transform that applies different scaling methods based on image modality.
    For CT images, it uses ScaleIntensityRanged with fixed intensity bounds.
    For MR images, it uses ScaleIntensityRangePercentilesd to scale based on intensity percentiles."""
    def __init__(self, keys: list, min: float = 0.0, max: float = 1.0):
        super().__init__(keys)
        self.min = min
        self.max = max
        self.ct_transform = ScaleIntensityRanged(
            keys=keys, a_min=-200, a_max=800, b_min=min, b_max=max, clip=True) # Scale CT images between -200 and 800 HU and map to [min, max]
        self.mr_transform = ScaleIntensityRangePercentilesd(
            keys=keys, lower=0.5, upper=99.5, b_min=min, b_max=max, 
            clip=True, relative=False, channel_wise=False) # Scale MR images based on 0.5 and 99.5 percentiles and map to [min, max]

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if d["modality"] == "ct":
            d = self.ct_transform(d)
        elif d["modality"] == "mr":
            d = self.mr_transform(d)
        else:
            raise ValueError(f"Unsupported modality: {d['modality']}")
        return d
    
class BiasFieldCorrection(MapTransform):
    """Apply N4 Bias Field Correction to the image."""
    def __init__(self, keys: list, shrink_factor: int = 1):
        super().__init__(keys)
        self.shrink_factor = shrink_factor

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if d.get("modality", None) != "mr":
            return d  # No correction for non-MR images
        
        for key in self.keys:
            original_tensor = d[key]
            img = d[key].numpy()  # convert to numpy array
            del_dim = 0
            while len(img.shape) > 3:
                img = np.squeeze(img, axis=0)  # remove singleton dimensions
                del_dim += 1
            img_sitk = sitk.GetImageFromArray(img.astype(np.float32))  # convert to SimpleITK image

            mask = sitk.OtsuThreshold(img_sitk, 0, 1, 200)  # create a mask using Otsu's method

            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50,50,30,20])  # set maximum number of iterations for each level

            shrink_factor = self.shrink_factor  # define a shrink factor to speed up processing
            img_shrink = sitk.Shrink(img_sitk, [shrink_factor]*img_sitk.GetDimension())
            mask_shrink = sitk.Shrink(mask, [shrink_factor]*mask.GetDimension())

            corrected_shrink = corrector.Execute(img_shrink, mask_shrink)  # apply N4 bias field correction to shrunken images
            log_bias_field = corrector.GetLogBiasFieldAsImage(img_sitk) # get the log bias field for original image
            img_corrected = img_sitk / sitk.Exp(log_bias_field)  # get the corrected image from original image resolution

            img_corrected = sitk.GetArrayFromImage(img_corrected)  # convert back to numpy array

            while del_dim > 0:
                img_corrected = np.expand_dims(img_corrected, axis=0)  # add back singleton dimensions
                del_dim -= 1

            d[key] = MetaTensor(
                img_corrected,
                meta=original_tensor.meta.copy(),
                dtype=original_tensor.dtype
            ) # Convert back to MetaTensor with original metadata

        return d
    
class BalancedCTMRSampler(Sampler):
    """A custom sampler that ensures alternating CT and MR modality between batches."""
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.ct_indices = [i for i, d in enumerate(dataset.data) if d['modality'] == 'ct']
        self.mr_indices = [i for i, d in enumerate(dataset.data) if d['modality'] == 'mr']
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.ct_indices)
            np.random.shuffle(self.mr_indices)

        for ct_idx, mr_idx in zip(self.ct_indices, self.mr_indices):
            yield ct_idx
            yield mr_idx

    def __len__(self):
        return min(len(self.ct_indices), len(self.mr_indices)) * 2