import os
import datetime
import glob
import pathlib
from enum      import Enum
from tqdm.auto import tqdm

import uuid

import numpy   as np
import pandas  as pd
import nibabel as nib

from data_declaration import *
from loader_helper    import loader_helper
from architecture     import *
from evaluation       import evaluate_model


import torch
import torch.nn    as nn
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

        
def save_weights(model_in, uuid, fold=1, task : Task = None):
    
    root_path = ""
    
    if (task == Task.CN_v_AD):
        root_path = "../weights/CN_v_AD/"     + uuid + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + uuid + "/"
    
    if fold == 1 : os.mkdir(root_path) #otherwise it already exists
    
    while (True):    
        
        s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format(fold, date=datetime.datetime.now())
        
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
        
        
    else: #must be camull       
        
        if (path == None):
            model = load_cam_model("../weights/camnet/fold_0_weights-2020-04-09_18_29_02")
        else:
            model = load_cam_model(path)
            

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
        

def train_camull(ld_helper, k_folds=5, model = None, epochs=40):

    task      = ld_helper.get_task()
    uuid_     = uuid.uuid4().hex
    model_cop = model

    for k_ind in range(k_folds):
        
        if (model_cop == None):
            model = build_arch()
        else:
            model = model_cop
   
        train_dl = ld_helper.get_train_dl(k_ind)
        train_loop(model, train_dl, epochs)
        save_weights(model, uuid_, fold=k_ind+1, task=task)
    
    return uuid_
    

def main():

    #CN v AD
    ld_helper = loader_helper(task=Task.CN_v_AD)
    uuid = train_camull(ld_helper, epochs=40)
    evaluate_model(device, uuid, ld_helper)

    #transfer learning for pMCI v sMCI
    ld_helper.change_task(Task.sMCI_v_pMCI)
    model = load_model("camull", uuid)
    uuid  = train_camull(ld_helper, model=model, epochs=40)
    evaluate_model(device, uuid, ld_helper)

    
main()