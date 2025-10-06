GLOBAL_LABELS = {
    0 : "background",
    1 : "BA",
    2 : "R-P1P2",
    3 : "L-P1P2",
    4 : "R-ICA",
    5 : "R-M1",
    6 : "L-ICA",
    7 : "L-M1",
    8 : "R-Pcom",
    9 : "L-Pcom",
    10 : "Acom",
    11 : "R-A1A2",
    12 : "L-A1A2",
    13 : "R-A3",
    14 : "L-A3",
    15 : "3rd-A2",
    16 : "3rd-A3",
    17 : "R-M2",
    18 : "R-M3",
    19 : "L-M2",
    20 : "L-M3",
    21 : "R-P3P4",
    22 : "L-P3P4",
    23 : "R-VA",
    24 : "L-VA",
    25 : "R-SCA",
    26 : "L-SCA",
    27 : "R-AICA",
    28 : "L-AICA",
    29 : "R-PICA",
    30 : "L-PICA",
    31 : "R-AChA",
    32 : "L-AChA",
    33 : "R-OA",
    34 : "L-OA",
    35 : "R-ECA",
    36 : "L-ECA",
    37 : "R-STA",
    38 : "L-STA",
    39 : "R-MaxA",
    40 : "L-MaxA",
    41 : "R-MMA",
    42 : "L-MMA",
    43 : "VoG",
    44 : "StS",
    45 : "ICVs",
    46 : "R-BVR",
    47 : "L-BVR",
    48 : "SSS"
}

CTA_TO_GLOBAL = {i : i for i in range(35)} | {35 : 43, 36 : 44, 37 : 45, 38 : 46, 39 : 47, 40 : 48}
MRA_TO_GLOBAL = {i : i for i in range(43)}

GLOBAL_TO_CTA = {v: k for k, v in CTA_TO_GLOBAL.items()}
GLOBAL_TO_MRA = {v: k for k, v in MRA_TO_GLOBAL.items()}

from monai.transforms import MapTransform
from monai.data import MetaTensor
import numpy as np
import torch

class RelabelByModality(MapTransform):
    """
    Apply mapping to input label map.  
    Origin : label = [(global) 1 -> 34 (CTA and MRA unique labels cofounded) 35 -> 42]  
    Our mapping : label = [(global) 1 -> 34 (MRA) 35 -> 42 (CTA) 43 -> 48]  
    """
    def __init__(self, keys, reverse = False):
        super().__init__(keys)
        self.reverse = reverse

    def __call__(self, data):
        d = dict(data)
        modality = d.get("modality", None)

        if modality == 'ct':
            mapping = GLOBAL_TO_CTA if self.reverse else CTA_TO_GLOBAL
        elif modality == 'mr':
            mapping = GLOBAL_TO_MRA if self.reverse else MRA_TO_GLOBAL
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        for key in self.keys:
            d[key] = self._remap(d[key], mapping)

        return d

    def _remap(self, arr, mapping):
        # Handle MetaTensor, torch.Tensor, or numpy array
        is_metatensor = isinstance(arr, MetaTensor)
        
        # Convert to numpy for mapping
        if isinstance(arr, (MetaTensor, torch.Tensor)):
            arr_np = arr.cpu().numpy() if arr.is_cuda else arr.numpy()
        else:
            arr_np = np.asarray(arr)
        
        # Create output array
        out = arr_np.copy()
        
        # Apply mapping
        for orig, target in mapping.items():
            out[arr_np == orig] = target
        
        # Restore original type
        if is_metatensor:
            return MetaTensor(out, meta=arr.meta)
        elif isinstance(arr, torch.Tensor):
            return torch.from_numpy(out).to(arr.device)
        else:
            return out