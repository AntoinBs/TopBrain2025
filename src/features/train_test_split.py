import os

import pandas as pd
import numpy as np

import nibabel as nib

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.utils.labels_mapping import RelabelByModality


def build_processed_df(data_dir: str) -> pd.DataFrame:
    images = []
    labels = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if not folder.startswith('images') and not folder.startswith('labels'):
            continue

        for file in os.listdir(folder_path):
            if not (file.endswith('.nii') or file.endswith('.nii.gz')):
                continue
            file_path = os.path.join(folder_path, file)
            file_name = file
            if folder.startswith('images'): # images
                modality = file.split('_')[1]
                images.append({'file_name': file_name, 'image_path': file_path, 'modality': modality})
            else:   # labels
                labels.append({'file_name': file_name, 'label_path': file_path})

    images_df = pd.DataFrame(images)
    labels_df = pd.DataFrame(labels)
    df = images_df.merge(labels_df, on='file_name', how='left')

    print(f"Total images found: {len(df)}")
    print(f"Total labels found: {df['label_path'].notnull().sum()}")

    return df

def build_label_presence_matrix(df: pd.DataFrame, label_path_col: str = 'label_path') -> np.ndarray:
    matrix = []
    for _, row in df.iterrows():
        label_map = nib.load(row[label_path_col]).get_fdata() if pd.notnull(row[label_path_col]) else None
        if label_map is not None:
            relabel_transform = RelabelByModality(keys=['label'], reverse=False)
            label_map = relabel_transform({'label': label_map, 'modality': row['modality']})['label']
            unique_labels = np.unique(label_map)
            presence_vector = np.zeros(49, dtype=int)  # Assuming labels range from 0 to 48
            for label in unique_labels:
                presence_vector[int(label)] = 1
        else:
            presence_vector = np.zeros(49, dtype=int)  # No labels present
        matrix.append(presence_vector)

    return np.array(matrix)

if __name__ == "__main__":

    df = build_processed_df('./data/processed')
    df_ct = df[df['modality'] == 'ct']
    df_mr = df[df['modality'] == 'mr']
    print("Building label presence matrix...")
    label_matrix_ct = build_label_presence_matrix(df_ct)
    label_matrix_mr = build_label_presence_matrix(df_mr)
    print("Splitting dataset into train and test sets...")

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_ct, test_idx_ct = next(msss.split(df_ct, label_matrix_ct))
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_mr, test_idx_mr = next(msss.split(df_mr, label_matrix_mr))
    train_df_ct = df_ct.iloc[train_idx_ct]
    test_df_ct = df_ct.iloc[test_idx_ct]
    train_df_mr = df_mr.iloc[train_idx_mr]
    test_df_mr = df_mr.iloc[test_idx_mr]
    
    train_df = pd.concat([train_df_ct, train_df_mr]).reset_index(drop=True)
    test_df = pd.concat([test_df_ct, test_df_mr]).reset_index(drop=True)

    print(f"Train set size: {len(train_df)}")
    print(f" - CT: {len(train_df_ct)}")
    print(f" - MR: {len(train_df_mr)}")
    print(f"Test set size: {len(test_df)}")
    print(f" - CT: {len(test_df_ct)}")
    print(f" - MR: {len(test_df_mr)}")

    print("Saving train and test splits...")
    train_df.to_csv('./data/processed/train_split.csv', index=False)
    test_df.to_csv('./data/processed/test_split.csv', index=False)
    print("Train and test splits have been saved to './data/processed'.")