from architecture     import load_cam_model
from data_declaration import Task

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import os
import glob
import datetime

import torch
import torch.nn    as nn
import torch.optim as optim

from tqdm.auto import tqdm

import enlighten

device = None
ticks = None
tocks = None
data_pbar = None

def make_folders():
    if (not os.path.exists("../logs/")):
        os.mkdir("../logs/")
    if (not os.path.exists("../graphs/")):
        os.mkdir("../graphs/")

def evaluate_model(device_in, uuid, ld_helper):

    global device
    global ticks
    global tocks
    global data_pbar

    device = device_in

    manager = enlighten.get_manager()
    ticks = manager.counter(total=5, desc='Fold', unit='folds')
    tocks = manager.counter(total=10, desc='Threshold-------------', unit='notches')
    data_pbar = manager.counter(total=0, desc='Data------------------', unit='batches')

    make_folders()
    log_path = "../logs/" + uuid + ".txt"

    if (os.path.exists(log_path)):
        filein     = open(log_path, 'a')
    else:
        filein     = open(log_path, 'w')

        
    task_str   = ld_helper.get_task_string()

    filein.write("\n")
    filein.write("==========================\n")
    filein.write("===== Log for camull =====\n")
    filein.write("==========================\n")
    filein.write("\n")
    filein.write("----- Date: {date:%Y-%m-%d_%H:%M:%S} -----\n".format(date=datetime.datetime.now()))
    filein.write("\n")
    filein.write("\n")

    tot_acc = 0; tot_sens = 0; tot_spec = 0; tot_roc_auc = 0
    fold = 0

    srch_path = "../weights/{}/".format(task_str) + uuid + "/*"
    for path in glob.glob(srch_path):

        print("Evaluating fold: ", fold + 1)

        model   = load_cam_model(path)
        model.to(device)
        test_dl = ld_helper.get_test_dl(fold)
        data_pbar.total = len(test_dl)

        if (not os.path.exists("../graphs/" + uuid)) : os.mkdir("../graphs/" + uuid)
        metrics = get_roc_auc(model, test_dl, figure=True, path = "../graphs/" + uuid, fold=fold+1)
        accuracy, sensitivity, specificity, roc_auc, you_max, you_thresh = [*metrics]

        print("Evaluated fold: {}".format(fold+1))
        
        filein.write("=====   Fold {}  =====".format(fold+1))
        filein.write("\n")
        filein.write("Threshold {}".format(you_thresh))
        filein.write("\n")
        filein.write("--- Accuracy     : {}\n".format(accuracy))
        filein.write("--- Sensitivity  : {}\n".format(sensitivity))
        filein.write("--- Specificity  : {}\n".format(specificity))
        filein.write("--- Youdens stat : {}\n".format(you_max))
        filein.write("\n")
        filein.write("(Variable Threshold)")
        filein.write("--- ROC AUC     : {}\n".format(roc_auc))
        filein.write("\n")

        tot_acc += accuracy; tot_sens += sensitivity; tot_spec += specificity; tot_roc_auc += roc_auc
        fold += 1
        ticks.update()
        tocks.count = 0


    avg_acc     =  (tot_acc     / 5)
    avg_sens    =  (tot_sens    / 5)
    avg_spec    =  (tot_spec    / 5)
    avg_roc_auc =  (tot_roc_auc / 5)

    filein.write("\n")
    filein.write("===== Average Across 5 folds =====")
    filein.write("\n")
    filein.write("--- Accuracy    : {}\n".format(avg_acc))
    filein.write("--- Sensitivity : {}\n".format(avg_sens))
    filein.write("--- Specificity : {}\n".format(avg_spec))
    filein.write("\n")
    filein.write("(Variable Threshold)")
    filein.write("--- ROC AUC     : {}\n".format(avg_roc_auc))
    filein.write("\n")


def get_roc_auc(model_in, test_dl, figure=False, path=None, fold=1):
    
    fpr = [] #1-specificity
    tpr = []

    youden_s_lst = []

    opt_acc = 0; opt_sens = 0; opt_spec = 0
    youdens_s_max = 0
    optimal_thresh = 0

    print("Walking through thresholds.")
    for t in range(0, 10, 1):

        thresh = t/10
        acc, sens, spec = get_metrics(model_in, test_dl, thresh)
        tpr.append(sens)
        fpr.append(1 - spec)


        youdens_s = sens + spec - 1

        if (youdens_s > youdens_s_max): 

            youdens_s_max = youdens_s; 
            optimal_thresh = thresh
            opt_acc = acc; opt_sens = sens; opt_spec = spec

        tocks.update()
        

    roc_auc = -1
    try:
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(e)
    metrics = [opt_acc, opt_sens, opt_spec, roc_auc, youdens_s_max, optimal_thresh]

    if(figure):

        if (path == None):
            path = "../graphs/auc-{date:%Y-%m-%d_%H-%M-%S}.png".format(date=datetime.datetime.now())
        else:
            #append dir
            path = path + "/auc-fold{}-{date:%Y-%m-%d_%H-%M-%S}.png".format(fold, date=datetime.datetime.now())
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fold {}'.format(fold))
        plt.legend(loc="lower right")
        plt.savefig(path)
    
    return metrics


def get_metrics(model_in, test_dl, thresh=0.5, param_count=False):
        
    correct = 0; total = 0
    model_in.eval()
    
    TP = 0.000001; TN = 0.000001; FP = 0.000001; FN = 0.000001
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(test_dl):
            
            batch_X  = sample_batched['mri'].to(device)
            batch_Xb = sample_batched['clin_t'].to(device)
            batch_y  = sample_batched['label'].to(device)
            
            for i in range(4): #hard coded batch size of 4
                
                real_class = batch_y[i].item()
                X = batch_X[i].view(-1, 1, 110, 110, 110)
                Xb = batch_Xb[i].view(1, 21)
                net_out = model_in((batch_X[i].view(-1, 1, 110, 110, 110), batch_Xb[i].view(1, 21)))
                predicted_class = 1 if net_out > thresh else 0
                
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

            data_pbar.update()
    
    accuracy = round(correct/total, 3)
    sensitivity = round((TP / (TP + FN)), 3)
    specificity = round((TN / (TN + FP)), 3)

    data_pbar.count = 0
    
    return (accuracy, sensitivity, specificity)
