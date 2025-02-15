'''The following module declares the Dataset objects required by torch to iterate over the data.'''
from enum import Enum
import glob
import pathlib

import numpy as np
import nibabel as nib
import pandas as pd

import torch
from torch.utils.data import Dataset


class Task(Enum):
    '''
        Enum class for the two classification tasks
    '''
    NC_v_AD = 1
    sMCI_v_pMCI = 2

    def __str__(self) -> str:
        if super().__str__() == "Task.NC_v_AD":
            return "NC_v_AD"
        else:
            return "sMCI_v_pMCI"

def get_im_id(path):
    '''Gets the image id from the file path string'''
    fname = path.stem
    im_id_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    im_id_str = fname.split("_")[-1][1:]
    return int(im_id_str)

def get_ptid(path):
    '''Gets the ptid from the file path string'''
    fname = path.stem
    ptid_str = ""
    count = 0
    for char in fname:
        if count == 4:
            break
        if 0 < count < 4:
            ptid_str += char

        if char == '_':
            count += 1

    return ptid_str[:-1]

def get_acq_year(im_data_id, im_df):
    '''Gets the acquisition year from a pandas dataframe by searching the image id'''
    acq_date = im_df[im_df['Image Data ID'] == im_data_id]["Acq Date"].iloc[0]
    acq_year_str = ""

    slash_count = 0
    for char in acq_date:
        if char == "/":
            slash_count += 1

        if slash_count == 2:
            acq_year_str += char

    return acq_year_str[1:]

def get_label(path, labels):
    '''Gets label from the path'''
    label_str = path.parent.stem
    label = None

    if label_str == labels[0]:
        label = np.array([0], dtype=np.double)
    elif label_str == labels[1]:
        label = np.array([1], dtype=np.double)
    return label

def get_mri(path):
    '''Gets a numpy array representing the mri object from a file path'''
    mri = nib.load(str(path)).get_fdata()
    mri = mri.reshape(1, 110, 110, 110)  # Using reshape instead of resize
    return mri.astype(np.float32)  # Ensure float32 type for consistency

def get_clinical(im_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    clinical = np.zeros(21)
    
    row = clin_df.loc[clin_df["Image Data ID"] == im_id]
    
    # Use iloc consistently for integer-based indexing
    for k in range(1, 22):
        clinical[k-1] = row.iloc[0].iloc[k]  # or use: row.iloc[0, k]
    
    return clinical


class MRIDataset(Dataset):
    '''Provides an object for the MRI data that can be iterated.'''
    def __init__(self, root_dir, labels, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.directories = []
        self.len = 0
        self.labels = labels
        self.clin_data = pd.read_csv("../data/clinical.csv")
        train_dirs = []

        for label in labels:
            train_dirs.append(root_dir + label)

        for train_dir in train_dirs:
            for path in glob.glob(train_dir + "/*"):
                self.directories.append(pathlib.Path(path))

        self.len = len(self.directories)

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            path = self.directories[idx]
            im_id = get_im_id(path)
            mri = get_mri(path)
            clinical = get_clinical(im_id, self.clin_data)
            label = get_label(path, self.labels)

            sample = {'mri': mri, 'clinical':clinical, 'label':label}

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise

class ToTensor():
    '''Convert ndarrays in sample to Tensors with proper normalization and type'''
    def __call__(self, sample):
        image, clinical, label = sample['mri'], sample['clinical'], sample['label']
        
        # Proper MRI normalization using z-score for non-zero voxels
        mask = image != 0
        mean = image[mask].mean()
        std = image[mask].std()
        normalized_mri = np.zeros_like(image, dtype=np.float64)  # Changed to float64
        normalized_mri[mask] = (image[mask] - mean) / (std + 1e-10)
        
        # Convert to tensors with double precision
        mri_t = torch.from_numpy(normalized_mri).double()  # Explicitly convert to double
        clin_t = torch.from_numpy(clinical).double()  # Explicitly convert to double
        label = torch.from_numpy(label).double()
        
        return {
            'mri': mri_t,
            'clin_t': clin_t,
            'label': label
        }
