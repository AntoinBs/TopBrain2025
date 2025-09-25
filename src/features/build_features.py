import pandas as pd
import os
import monai

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    SaveImaged
)

from utils import CustomScaleIntensityRanged, BiasFieldCorrection
from monai.data import DataLoader, Dataset

def custom_filename(metadict: dict, saver: monai.transforms.Transform) -> str:
    """
    Custom function to generate filenames based on the original filename.
    This function can be modified to fit specific naming conventions.
    """
    subject = (
        metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
        if metadict
        else getattr(saver, "_data_index", 0)
    )
    original_filename = os.path.basename(subject)
    subject = "_".join(original_filename.split('.')[0].split('_')[:3]) + "." + ".".join(original_filename.split('.')[1:])
    patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
    return {"subject": f"{subject}", "idx": patch_index}

def build_metadata_dataframe():
    """
    Build a dataframe containing metadata for all images and labels in the raw data directory.
    The dataframe will have columns: 'file_name', 'file_path', 'modality', 'label_path'.
    """
    metadatas = []

    for folder in os.listdir('./data/raw'): # iterate over all folders in raw data directory to construct images dataframe
        folder_path = os.path.join('./data/raw', folder)
        if not os.path.isdir(folder_path): # skip if not a directory
            continue
        if folder.startswith('itksnap') or folder.startswith('labels'): # skip labels files
            continue

        print(f"Extracting images paths from folder: {folder_path}")

        for file in os.listdir(folder_path):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(folder_path, file)
                file_name = file

                modality = file.split('_')[1]

                metadatas.append({'file_name': file_name, 'file_path': file_path, 'modality': modality})

    for folder in os.listdir('./data/raw'):
        if not folder.startswith('labels'):
            continue
        folder_path = os.path.join('./data/raw', folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Extracting labels paths from folder: {folder_path}")

        for file in os.listdir(folder_path):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(folder_path, file)
                file_name = file

                for mt in metadatas:
                    if mt['file_name'].split('.')[0].split('_')[:3] == file_name.split('.')[0].split('_')[:3]:
                        mt['label_path'] = file_path

    df = pd.DataFrame(metadatas)

    print(f"Total images found: {len(df)}")
    print(f"Total labels found: {df['label_path'].notnull().sum()}")

    return df

if __name__ == "__main__":
    df = build_metadata_dataframe()
    data_dict = []

    for idx, row in df.iterrows():
        data_dict.append({"image": row['file_path'], "label": row['label_path'], "modality": row['modality']})

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=['image', 'label']),
        BiasFieldCorrection(keys=['image'], shrink_factor=4),
        CustomScaleIntensityRanged(keys=['image'], min=0.0, max=1.0),
        Orientationd(keys=['image', 'label'], axcodes="LPS"),
        Spacingd(keys=['image', 'label'], pixdim=(0.7, 0.7, 0.7), mode=['bilinear', 'nearest']),
        SpatialPadd(keys=['image', 'label'], spatial_size=(96, 96, 96)),
        SaveImaged(
            keys=['image'],
            output_dir='./data/preprocessed/images',
            output_name_formatter=custom_filename,
            output_postfix='preproc',
            separate_folder=False
        ),
        SaveImaged(
            keys=['label'],
            output_dir='./data/preprocessed/labels',
            output_postfix='preproc',
            separate_folder=False
        )
    ])

    ds = Dataset(data=data_dict, transform=transforms)
    dataloader = DataLoader(ds, batch_size=1)

    print("Starting preprocessing and saving images and labels...")

    l = len(dataloader)
    for i, batch in enumerate(dataloader):
        print(f"Image {i+1}/{l} saved")
        print(f"Label {i+1}/{l} saved")

    print("Preprocessing and saving completed.")
    print("Preprocessed images and labels are saved in './data/preprocessed/images' and './data/preprocessed/labels' respectively.")