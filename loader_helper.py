from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import Subset

from data_declaration import *

from enum import Enum



class loader_helper:
    
    def __init__(self, task : Task = Task.CN_v_AD):
        
        self.task = task
        self.labels = []

        if (task == Task.CN_v_AD):
            self.labels = ["CN", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(root_dir="../data/",
                 labels=self.labels,
                transform=transforms.Compose([
                    ToTensor()
                ]))

        self.indices = []
        self.set_indices()


    def get_task(self):
        return self.task


    def get_task_string(self):

        if (self.task == Task.CN_v_AD):
            return "CN_v_AD"
        else:
            return "sMCI_v_pMCI"


    def change_ds_labels(self, labels_in):
        
        self.dataset = MRIDataset(root_dir     = "../data/",
                                  labels       = labels_in,
                                  transform    = transforms.Compose([
                                      ToTensor()])
                                 )

    
    def change_task(self, task: Task):

        self.task = task
        
        if (task == Task.CN_v_AD):
            self.labels = ["CN", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(root_dir     = "../data/",
                            labels       = self.labels,
                            transform    = transforms.Compose([
                                ToTensor()])
                            )

        self.set_indices()

        
    def set_indices(self, total_folds=5):

        test_split      = .2
        shuffle_dataset = True
        random_seed     = 42

        dataset_size = len(self.dataset)
        indices      = list(range(dataset_size))
        split        = int(np.floor(test_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        fold_indices = []
        lb_split = 0
        ub_split = split

        for k in range(total_folds):
            train_indices = indices[:lb_split] + indices[ub_split:] 
            test_indices  = indices[lb_split:ub_split]
            lb_split      = split
            ub_split      = 2*split
            fold_indices.append((train_indices, test_indices))

        self.indices = fold_indices


    def make_loaders(self, shuffle=True):

        fold_indices = self.get_indices()

        for k in range(5):

            train_ds = Subset(self.dataset, fold_indices[k][0])
            test_ds  = Subset(self.dataset, fold_indices[k][1])

            train_dl = DataLoader(train_ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)
            test_dl  = DataLoader(test_ds,  batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

        print(len(test_ds))

        return (train_dl_lst, test_dl_lst)

    
    def get_train_dl(self, fold_ind, shuffle=True):

        train_ds = Subset(self.dataset, self.indices[fold_ind][0])
        train_dl = DataLoader(train_ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

        return train_dl


    def get_test_dl(self, fold_ind, shuffle=True):

        test_ds = Subset(self.dataset, self.indices[fold_ind][1])
        test_dl = DataLoader(test_ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

        return test_dl



