import os
from enum import Enum
import glob
import pathlib

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Task(Enum):
    CN_v_AD = 1
    sMCI_v_pMCI = 2

    
def get_im_id(path):

    fname = path.stem
    im_id_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    im_id_str = fname.split("_")[-1][1:] 
    return int(im_id_str)
    
    
def get_ptid(path):

    fname = path.stem
    ptid_str = ""

    count = 0     
    for char in fname:
        if (count == 4):
            break
        elif (count  > 0 and count < 4):
            ptid_str += char

        if (char == '_'):
            count += 1

    return ptid_str[:-1]
    
    
def get_acq_year(im_data_id, im_df):

    #get the acq year
    #there will only ever be one record output
    acq_date = im_df[im_df['Image Data ID'] == im_data_id]["Acq Date"].iloc[0] 
    acq_year_str = ""

    slash_count = 0
    for c in acq_date:
        if (c == "/"):
            slash_count += 1

        if (slash_count == 2):
            acq_year_str += c

    return acq_year_str[1:]


def get_label(path, labels):
    
    label_str = path.parent.stem
    label     = None

    if (label_str == labels[0]):
        label = np.array([0], dtype=np.double)
    elif (label_str == labels[1]):
        label = np.array([1], dtype=np.double)
        
    return label


def get_mri(path):

    mri = nib.load(str(path)).get_fdata()
    mri.resize(1,110,110,110)
    mri = np.asarray(mri)

    return mri


def get_clinical(im_id, clin_df):

    clinical = np.zeros(21)

    row     = clin_df.loc[clin_df["Image Data ID"] == im_id]

    for k in range(1, 22):
        clinical[k-1] = row.iloc[0][k]

    return clinical



class MRIDataset(Dataset):
    
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
        
        for dir in train_dirs:
            for path in glob.glob(dir + "/*"):
                self.directories.append(pathlib.Path(path))
                
        self.len = len(self.directories)
        
        
    def __len__(self):
        
        return self.len 
    
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        repeat = True                 
    
        while(repeat):
            try:
                path     = self.directories[idx]
                im_id    = get_im_id(path)
                mri      = get_mri(path)
                clinical = get_clinical(im_id, self.clin_data)
                label    = get_label(path, self.labels)

                sample = {'mri': mri, 'clinical':clinical, 'label':label}

                if self.transform:
                    sample = self.transform(sample)

                return sample 

            except Exception as e:
                #print(e)
                if (idx < self.len): 
                    idx += 1
                else:
                    idx = 0
                    
        return sample

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, clinical, label = sample['mri'], sample['clinical'], sample['label']
        mri_t = torch.from_numpy(image) / 255.0
        clin_t = torch.from_numpy(clinical)
        label = torch.from_numpy(label).double()
        return {'mri': mri_t,
                'clin_t': clin_t,
                'label': label}
