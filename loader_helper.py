'''The following module deals with creating the loader he'''
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

import numpy as np

from data_declaration import MRIDataset, Task, get_ptid
from data_declaration import ToTensor
from collections import defaultdict

class LoaderHelper:
    '''An abstract class for assisting with dataset creation.'''
    def __init__(self, task: Task = Task.NC_v_AD):

        self.task = task
        self.labels = []

        if task == Task.NC_v_AD:
            self.labels = ["NC", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(root_dir="../data/",
                labels=self.labels,
                transform=transforms.Compose([
                ToTensor()
            ]))
        self.indices = []
        self.set_indices()

    def print_split_stats(self, fold_idx):
        """Print detailed statistics about the split"""
        train_indices, val_indices = self.indices[fold_idx]
        
        # Get patient IDs
        train_patients = set(get_ptid(self.dataset.directories[i]) for i in train_indices)
        val_patients = set(get_ptid(self.dataset.directories[i]) for i in val_indices)
        
        print(f"\nFold {fold_idx} Patient Statistics:")
        print(f"Training patients: {len(train_patients)}")
        print(f"Validation patients: {len(val_patients)}")
        print(f"Patient overlap: {len(train_patients & val_patients)}")  # Should be 0

    def print_dataset_stats(self):
        """Print distribution of classes in the dataset"""
        print("\nDataset Statistics:")
        total_samples = len(self.dataset)
        class_counts = {}
        
        for idx in range(total_samples):
            sample = self.dataset[idx]  # Get the full sample
            label = sample['label']  # Access label from dictionary
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = label[0]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print(f"Total samples: {total_samples}")
        for label, count in class_counts.items():
            class_name = self.labels[int(label)]
            print(f"Class {class_name}: {count} samples ({count/total_samples*100:.2f}%)")

    def print_fold_stats(self, fold_idx):
        """Print class distribution for a specific fold"""
        print(f"\nFold {fold_idx + 1} Statistics:")
        
        # Get train and val indices for this fold
        train_indices = self.indices[fold_idx][0]
        val_indices = self.indices[fold_idx][1]
        
        # Count classes in training set
        train_class_counts = {}
        for idx in train_indices:
            sample = self.dataset[idx]  # Get the full sample
            label = sample['label']  # Access label from dictionary
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = label[0]
            train_class_counts[label] = train_class_counts.get(label, 0) + 1
        
        # Count classes in validation set
        val_class_counts = {}
        for idx in val_indices:
            sample = self.dataset[idx]
            label = sample['label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = label[0]
            val_class_counts[label] = val_class_counts.get(label, 0) + 1
        
        print("Training set:")
        for label, count in train_class_counts.items():
            class_name = self.labels[int(label)]
            print(f"Class {class_name}: {count} samples ({count/len(train_indices)*100:.2f}%)")
        
        print("\nValidation set:")
        for label, count in val_class_counts.items():
            class_name = self.labels[int(label)]
            print(f"Class {class_name}: {count} samples ({count/len(val_indices)*100:.2f}%)")

    def get_task(self):
        '''gets task'''
        return self.task


    def get_task_string(self):
        '''Gets task string'''
        if self.task == Task.NC_v_AD:
            return "NC_v_AD"
        else:
            return "sMCI_v_pMCI"


    def change_ds_labels(self, labels_in):
        '''Function to change the labels of the dataset obj.'''
        self.dataset = MRIDataset(root_dir="../data/",
                                  labels=labels_in,
                                  transform=transforms.Compose([
                                      ToTensor()])
                                 )


    def change_task(self, task: Task):
        '''Function to change task of the Datasets'''
        self.task = task
        
        if (task == Task.NC_v_AD):
            self.labels = ["NC", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(root_dir="../data/",
                            labels=self.labels,
                            transform=transforms.Compose([
                                ToTensor()])
                            )

        self.set_indices()


    def set_indices(self, total_folds=5):
        # Get patient IDs for each scan
        patient_ids = [get_ptid(path) for path in self.dataset.directories]
        unique_patients = list(set(patient_ids))
        
        # Create patient to indices mapping
        patient_to_indices = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            patient_to_indices[pid].append(idx)
            
        # Get one label per patient
        patient_info = []
        patient_labels = []  # Separate list just for labels
        for pid in unique_patients:
            idx = patient_to_indices[pid][0]
            label = self.dataset[idx]['label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            n_scans = len(patient_to_indices[pid])
            patient_info.append((pid, label, n_scans))
            patient_labels.append(label)  # Add just the label
        
        # Sort by class and number of scans to ensure balanced splits
        patient_info.sort(key=lambda x: (x[1], x[2]))
        
        # Split using only the labels
        skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
        self.indices = []
        
        for train_idx, val_idx in skf.split(unique_patients, patient_labels):  # Use patient_labels here
            train_patients = [unique_patients[i] for i in train_idx]
            val_patients = [unique_patients[i] for i in val_idx]
            
            # Get all indices for each split
            train_indices = []
            val_indices = []
            for pid in train_patients:
                train_indices.extend(patient_to_indices[pid])
            for pid in val_patients:
                val_indices.extend(patient_to_indices[pid])
                
            self.indices.append((train_indices, val_indices))

    def make_loaders(self, shuffle=True):
        '''Makes the loaders'''
        fold_indices = self.indices()

        for k in range(5):

            train_ds = Subset(self.dataset, fold_indices[k][0])
            test_ds  = Subset(self.dataset, fold_indices[k][1])

            train_dl = DataLoader(train_ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)
            test_dl = DataLoader(test_ds,  batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

        print(len(test_ds))

        return (train_dl, test_dl)

    
    def get_train_dl(self, fold_ind, shuffle=True):
        train_ds = Subset(self.dataset, self.indices[fold_ind][0])
        
        # Calculate sample weights
        labels = [self.dataset[i]['label'].item() for i in self.indices[fold_ind][0]]
        class_counts = np.bincount([int(l) for l in labels])
        weights = 1.0 / class_counts[[int(l) for l in labels]]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        return DataLoader(
            train_ds,
            batch_size=4,
            sampler=sampler,  # Use weighted sampler instead of shuffle
            num_workers=4,
            drop_last=True,
            multiprocessing_context='spawn',
            persistent_workers=True
        )

    def get_val_dl(self, fold_ind, shuffle=True):
        val_ds = Subset(self.dataset, self.indices[fold_ind][1])
        val_dl = DataLoader(
            val_ds, 
            batch_size=4, 
            shuffle=shuffle, 
            num_workers=4,
            drop_last=True,
            multiprocessing_context='spawn',
            persistent_workers=True
        )
        return val_dl

    # Keep the test_dl method for final evaluation
    def get_test_dl(self, fold_ind, shuffle=True):
        return self.get_val_dl(fold_ind, shuffle)  # Just alias it for now



