from data_declaration import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import Subset

labels = ["CN", "AD"]

dataset = MRIDataset(root_dir="../data4/alldata/",
                 labels=labels,
                transform=transforms.Compose([
                    ToTensor()
                ]))

class loader_helper:
    
    def __init__(self):
        
        self.labels = ["CN", "AD"]
        self.dataset = MRIDataset(root_dir="../data4/alldata/",
                 labels=self.labels,
                transform=transforms.Compose([
                    ToTensor()
                ]))
    
    
    def change_ds_labels(self, labels_in):
        
        self.dataset = MRIDataset(root_dir     = "../data4/alldata/",
                                  labels       = labels_in,
                                  transform    = transforms.Compose([
                                      ToTensor()])
                                 )
    
    
    def get_indices(self, total_folds=5):

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

        return fold_indices


    def make_loaders(self, train_indices, test_indices):

        train_ds = Subset(self.dataset, train_indices)
        test_ds  = Subset(self.dataset, test_indices)

        train_dl = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
        test_dl  = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
        print(len(test_ds))

        return (train_dl, test_dl)
