'''The following module deals with creating the loader he'''
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

import numpy as np

from data_declaration import MRIDataset, Task, get_ptid
from data_declaration import ToTensor


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
        """Set indices using patient-level stratified k-fold"""
        from sklearn.model_selection import StratifiedKFold
        random_seed = 42

        # Access directories from the dataset instance
        directories = self.dataset.directories  # Make sure this attribute exists in MRIDataset
        
        # First, group indices by patient
        patient_groups = {}
        for idx, path in enumerate(directories):
            ptid = get_ptid(path)  # Use your existing get_ptid function
            if ptid not in patient_groups:
                patient_groups[ptid] = {
                    'indices': [],
                    'label': None
                }
            # Add this scan's index to the patient's group
            patient_groups[ptid]['indices'].append(idx)
            # Set the label if not already set
            if patient_groups[ptid]['label'] is None:
                sample = self.dataset[idx]
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                patient_groups[ptid]['label'] = label

        # Prepare patient-level data for stratification
        patient_ids = list(patient_groups.keys())
        patient_labels = [patient_groups[pid]['label'] for pid in patient_ids]

        # Print dataset-level statistics
        print("\nPatient-Level Class Distribution:")
        unique_labels, counts = np.unique(patient_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Class {self.labels[int(label)]}: {count} patients ({count/len(patient_labels)*100:.2f}%)")

        # Create stratified folds at patient level
        skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=random_seed)
        
        self.indices = []
        for train_idx, val_idx in skf.split(patient_ids, patient_labels):
            # Convert patient-level splits to scan-level indices
            train_patients = [patient_ids[i] for i in train_idx]
            val_patients = [patient_ids[i] for i in val_idx]
            
            train_indices = []
            val_indices = []
            
            # Get all scan indices for each patient
            for pid in train_patients:
                train_indices.extend(patient_groups[pid]['indices'])
            for pid in val_patients:
                val_indices.extend(patient_groups[pid]['indices'])

            # Print fold statistics
            print(f"\nFold Statistics:")
            print(f"Training set: {len(train_patients)} patients, {len(train_indices)} scans")
            print(f"Validation set: {len(val_patients)} patients, {len(val_indices)} scans")
            
            train_labels = [patient_groups[pid]['label'] for pid in train_patients]
            val_labels = [patient_groups[pid]['label'] for pid in val_patients]
            
            # Print class distribution for this fold
            print("Training set class distribution:")
            train_unique, train_counts = np.unique(train_labels, return_counts=True)
            for label, count in zip(train_unique, train_counts):
                print(f"Class {self.labels[int(label)]}: {count} patients ({count/len(train_labels)*100:.2f}%)")
            
            print("Validation set class distribution:")
            val_unique, val_counts = np.unique(val_labels, return_counts=True)
            for label, count in zip(val_unique, val_counts):
                print(f"Class {self.labels[int(label)]}: {count} patients ({count/len(val_labels)*100:.2f}%)")
            
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
        train_dl = DataLoader(
            train_ds, 
            batch_size=4, 
            shuffle=shuffle, 
            num_workers=4,
            drop_last=True,
            multiprocessing_context='spawn',
            persistent_workers=True
        )
        return train_dl

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



