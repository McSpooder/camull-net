from loader_helper import loader_helper
from architecture import load_cam_model

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import os
import glob

import torch
import torch.nn    as nn
import torch.optim as optim


def evaluate_model(device, uuid, ld_helper):

    filein = open("log.txt", 'a')

    fold = 0

    filein.write("\n")
    filein.write("==========================\n".format(fold + 1))
    filein.write("===== Log for camull =====\n".format(fold + 1))
    filein.write("==========================\n".format(fold + 1))
    filein.write("\n")
    filein.write("\n")

    tot_acc = 0; tot_sens = 0; tot_spec = 0; tot_roc_auc = 0

    for path in glob.glob("../weights/CN_v_AD/" + uuid + "/*"):
        
        print("Evaluating fold: ", fold + 1)

        model   = load_cam_model(path)
        test_dl = ld_helper.get_test_dl(fold)
        metrics = get_fold_metrics(model, test_dl)

        accuracy, sensitivity, specificity = [*metrics.values()]

        if (fold == 0) : os.mkdir("../graphs/" + uuid)
        metrics["roc_auc"] = get_roc_auc(model, test_dl, figure=True, path = "../graphs/" + uuid, fold=fold+1)
        
        filein.write("=====   Fold {}  =====".format(fold+1))
        filein.write("\n")
        filein.write("Threshold 0.5")
        filein.write("--- Accuracy    : {}\n".format(metrics["accuracy"]))
        filein.write("--- Sensitivity : {}\n".format(metrics["sensitivity"]))
        filein.write("--- Specificity : {}\n".format(metrics["specificity"]))
        filein.write("\n")
        filein.write("(Variable Threshold)")
        filein.write("--- ROC AUC     : {}\n".format(metrics["roc_auc"]))
        filein.write("\n")

        tot_acc += accuracy; tot_sens += sensitivity; tot_spec += specificity; tot_roc_auc += roc_auc
        fold += 1

    avg_acc     =  (tot_acc     / 5)  *  100
    avg_sens    =  (tot_sens    / 5)  *  100
    avg_spec    =  (tot_spec    / 5)  *  100
    avg_roc_auc =  (tot_roc_auc / 5)  *  100 

    filein.write("\n")
    filein.write("===== Average Across 5 folds =====")
    filein.write("\n")
    filein.write("Threshold 0.5")
    filein.write("--- Accuracy    : {}\n".format(avg_acc))
    filein.write("--- Sensitivity : {}\n".format(avg_sens))
    filein.write("--- Specificity : {}\n".format(avg_spec))
    filein.write("\n")
    filein.write("(Variable Threshold)")
    filein.write("--- ROC AUC     : {}\n".format(metrics["roc_auc"]))
    filein.write("\n")


def get_fold_metrics(model_in, test_dl, param_count=False):

        accuracy, sensitivity, specificity = get_metrics(model_in, test_dl, thresh=0.5)

        metrics = {}

        metrics["accuracy"]    = accuracy
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = specificity

        return metrics


def get_roc_auc(model_in, test_dl, figure=False, path=None, fold=1):
    
    fpr = [] #1-specificity
    tpr = []

    for t in range(0, 10, 1):
        thresh = t/10
        _, sens, spec = get_metrics(model_in, test_dl, thresh)
        tpr.append(sens)
        fpr.append(1 - spec)

    roc_auc = auc(fpr, tpr)

    if(figure):

        if (path == None):
            path = "../graphs/auc-{date:%Y-%m-%d_%H:%M:%S}.png".format(date=datetime.datetime.now())
        else:
            #append dir
            path = path + "/auc-fold{}".format(fold)
        
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

    return roc_auc


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
    
    accuracy = round(correct/total, 3)
    sensitivity = round((TP / (TP + FN)), 3)
    specificity = round((TN / (TN + FP)), 3)
    
    return (accuracy, sensitivity, specificity)
