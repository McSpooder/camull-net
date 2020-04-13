import os
import numpy as np
import nibabel as nib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import glob
import pandas as pd
import pathlib

from exp.data_declaration import *
from exp.models import *
from exp.camnet import *
from exp.loader_helper import loader_helper

import torch
import torch.nn as nn
import datetime

import torch.optim as optim

from enum import Enum

from sklearn.metrics import roc_curve, auc

ld_helper = loader_helper()

EPOCHS = 0
optimizer = None

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
        
         
            
class Task(Enum):
    CN_v_AD = 1
    sMCI_v_pMCI = 2
    

def evaluate_model(model_in, test_dl, thresh=0.5, param_count=False):
    
    if (param_count):
        
        total_params = sum(p.numel() for p in model_in.parameters())
        print("Total number of parameters is: ", total_params)
        
        total_trainable_params = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
        print("Total number of trainable parameters is: ", total_trainable_params)
        
    
    correct = 0
    total = 0
    model_in.eval()
    
    TP = 0.000001
    TN = 0.000001
    FP = 0.000001
    FN = 0.000001
    
#    print("Evaluate model for thresh {}".format(thresh))
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(test_dl):
            
            batch_X  = sample_batched['mri'].to(device)
            batch_Xb = sample_batched['clin_t'].to(device)
            batch_y  = sample_batched['label'].to(device)
            
            for i in range(4):
                
                real_class = batch_y[i].item()
                net_out = model_in((batch_X[i].view(-1, 1, 110, 110, 110), batch_Xb[i].view(1, 21)))
                predicted_class = 1 if net_out > thresh else 0
                
#                 print("--- net_out is: {}".format(predicted_class))
                
                if (predicted_class == real_class):
                    correct += 1
                    if (real_class == 0):
                        TN += 1
                    elif (real_class == 1):
                        TP += 1
                else:
                    if (real_class == 0):
                        FP += 1
                    elif (real_class == 1):
                        FN += 1
                    
                    
                total += 1
    
    accuracy = round(correct/total, 3)
    sensitivity = round((TP / (TP + FN)), 3)
    specificity = round((TN / (TN + FP)), 3)
    
#     print("--- The sensitivity is: {}".format(sensitivity))
#     print("--- The specificity is: {}".format(specificity))
#     print("")
    
    return (accuracy, sensitivity, specificity)


def generate_auc(fpr, tpr, roc_auc):
    
    s_path = "../graphs/auc-{date:%Y-%m-%d_%H:%M:%S}.png".format( date=datetime.datetime.now())
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('books_read.png')


def get_metrics(model_in, test_dl, gen_auc=False, param_count=False):
    
        accuracy, sensitivity, specificity = evaluate_model(model_in, test_dl, thresh=0.5)
        
        if (gen_auc == True):
            
            fpr = [] #1-specificity
            tpr = []
        
            for t in range(0, 10, 1):
                thresh = t/10
                _, sens, spec = evaluate_model(model_in, test_dl, thresh)
                tpr.append(sens)
                fpr.append(1 - spec)


            roc_auc = auc(fpr, tpr)
            print("TPR rate list is: ", tpr)
            print("FPR list is: ", fpr)
            generate_auc(fpr, tpr, roc_auc)
            
        else:
            roc_auc = -1

        return (accuracy, sensitivity, specificity, roc_auc)
        
        
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

    
    
def train_loop(model_in, model_name, train_dl, epochs):

    optimizer = None
    
    if (model_name == "vox"):
        optimizer = optim.Adam(model_in.parameters(), lr=27e-6)
    else:
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

        
def build_arch(use_arch="vox"):

    net = None

    if (use_arch == "vox"):  
        net = VoxCNN()
    else:
        net = Camnet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)
    net.double()
    
    return net

        
def train_model(model=None, model_name="vox", epochs=150, task: Task = None):
    
    k_folds = 5

    indices = ld_helper.get_indices()
    filein = open("log.txt", 'a')
    filein.write("===== Log for {} =====\n".format(model_name))
    filein.write("\n")
    
    model_cop = model

    for k in range(k_folds):

        filein.write("Training model on fold {}\n".format(k))
        train_dl, test_dl = ld_helper.make_loaders(indices[k][0], indices[k][1])
        
        if (model_cop == None):
            model = build_arch(use_arch=model_name)
        else:
            model = model_cop
        
        train_loop(model, model_name, train_dl, epochs)
        save_weights(model, model_name, fold=k, task=task)

        accuracy, sensitivity, specificity, roc_auc = get_metrics(model, test_dl, gen_auc=True)

        filein.write("--- Accuracy    : {}\n".format(accuracy))
        filein.write("--- Sensitivity : {}\n".format(sensitivity))
        filein.write("--- Specificity : {}\n".format(specificity))
        filein.write("--- AUC         : {}\n".format(roc_auc))
        filein.write("\n")
        
     #calculate average model metrics
    
    filein.close()
    
    
    
def main():
    
#     train camnet
#     ld_helper.change_ds_labels(["sMCI", "pMCI"])

      #########################
      ##### Training code #####
      #########################

          ##### AD vs CN      #####

#     train_model(model_name="camnet", epochs=40)

#     indices = ld_helper.get_indices()
#     train_dl, test_dl = ld_helper.make_loaders(indices[0][0], indices[0][1])
    
#     model = load_model("camnet")

        ##### pMCI vs sMCI #####
    
    
    indices           = ld_helper.get_indices()
    train_dl, test_dl = ld_helper.make_loaders(indices[0][0], indices[0][1])
    
    model = load_model("camnet", "../weights/camnet/fold_2_weights-2020-04-12_15:48:22")
    
    ld_helper.change_ds_labels(["sMCI", "pMCI"])
    train_model(model, epochs=40, task=Task.sMCI_v_pMCI)

    ###########################
    ##### Evaluation code #####
    ###########################
    
        ##### pMCI vs sMCI #####
    accuracy, sensitivity, specificity = evaluate_model(model, test_dl, param_count=True)

#     #Evaluate on AD vs CN
#     accuracy, sensitivity, specificity = evaluate_model(model, test_dl, param_count=True)
#     accuracy, sensitivity, specificity, roc_auc = get_metrics(model, test_dl, gen_auc=True, param_count=False)
#     print("The accuracy is: ", accuracy)
#     print("The sensitivity is: ", sensitivity)
#     print("The specificity is: ", specificity)
    
#     #Evaluate on sMCI vs pMCI
#     ld_helper.change_ds_labels(["sMCI", "pMCI"])

    
    
    
main()