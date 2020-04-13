import os
from enum import Enum
import datetime
import glob
import pathlib
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import nibabel as nib

from data_declaration import *
from loader_helper import loader_helper
from architecture import *
import evaluation


import torch
import torch.nn as nn
import torch.optim as optim



ld_helper = loader_helper()

EPOCHS = 0
optimizer = None

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
        
         
            
class Task(Enum):
    CN_v_AD = 1
    sMCI_v_pMCI = 2
        
        
def save_weights(model_in, model_name, fold=0, task : Task = None):
    
    root_path = ""
    
    if (task == Task.CN_v_AD):
        root_path = "../weights/CN_v_AD/" + model_name + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + model_name + "/"
    
    
    while (True):    
        
        s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format( date=datetime.datetime.now())
        
        if (os.path.exists(s_path)):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break
            
            
            
def load_model(arch, path=None):
    
    if (arch == "vox"):
        
        if (path == None):         
            model = get_model("./weights/vox_arch_weights-2020-03-18_13:46:41")
        else:            
            model = get_model(path)
        
        
    else: #must be camnet       
        
        if (path == None):
            model = load_camnet_model("../weights/camnet/fold_0_weights-2020-04-09_18_29_02")
        else:
            model = load_camnet_model(path)
            

    return model

        
def build_arch():

    net = Camull()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)
    net.double()
    
    return net


def train_loop(model_in, train_dl, epochs):

    optimizer = optim.Adam(model_in.parameters(), lr=0.001, weight_decay=5e-5)
        
    loss_function = nn.BCELoss()
    
    model_in.train()

    for i in range (epochs):

        for i_batch, sample_batched in enumerate(tqdm(train_dl)):         

            batch_X  = sample_batched['mri'].to(device)
            batch_Xb = sample_batched['clin_t'].to(device)
            batch_y  = sample_batched['label'].to(device)
            
            model_in.zero_grad()
            outputs = model_in((batch_X, batch_Xb))

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss.item(), 5)))
        

def train_camull(model=None, epochs=150, task: Task = None):
    
    k_folds = 5
    indices = ld_helper.get_indices()

    filein = open("log.txt", 'a')
    filein.write("===== Log for camull =====\n")
    filein.write("\n")
    
    model_cop = model

    for k in range(k_folds):

        filein.write("Training model on fold {}\n".format(k))
        train_dl, test_dl = ld_helper.make_loaders(indices[k][0], indices[k][1])
        
        if (model_cop == None):
            model = build_arch()
        else:
            model = model_cop
        
        train_loop(model, train_dl, epochs)
        save_weights(model, fold=k, task=task)

        accuracy, sensitivity, specificity, roc_auc = get_metrics(model, test_dl, gen_auc=True)

        filein.write("--- Accuracy    : {}\n".format(accuracy))
        filein.write("--- Sensitivity : {}\n".format(sensitivity))
        filein.write("--- Specificity : {}\n".format(specificity))
        filein.write("--- AUC         : {}\n".format(roc_auc))
        filein.write("\n")
        
     #calculate average model metrics
    
    filein.close()
    
    
    
def main():
    
    #indices           = ld_helper.get_indices()
    #train_dl, test_dl = ld_helper.make_loaders(indices[0][0], indices[0][1])
    
    #model = load_model("camnet", "../weights/camnet/fold_2_weights-2020-04-12_15:48:22")
    
    #ld_helper.change_ds_labels(["sMCI", "pMCI"])
    train_camull(epochs=40, task=Task.CN_v_AD)

#   accuracy, sensitivity, specificity, roc_auc = evaluate_model(model, test_dl, param_count=True) 
    
main()