import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def generate_auc(fpr, tpr, roc_auc):
    
    s_path = "../graphs/auc-{date:%Y-%m-%d_%H:%M:%S}.png".format(date=datetime.datetime.now())
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(s_path)


def evaluate_model(model_in, test_dl, gen_auc=False, param_count=False):
    
        accuracy, sensitivity, specificity = get_metrics(model_in, test_dl, thresh=0.5)
        
        if (gen_auc == True):
            
            fpr = [] #1-specificity
            tpr = []
        
            for t in range(0, 10, 1):
                thresh = t/10
                _, sens, spec = get_metrics(model_in, test_dl, thresh)
                tpr.append(sens)
                fpr.append(1 - spec)

            roc_auc = auc(fpr, tpr)
            print("TPR rate list is: ", tpr)
            print("FPR list is: ", fpr)
            generate_auc(fpr, tpr, roc_auc)
            
        else:
            roc_auc = -1

        return (accuracy, sensitivity, specificity, roc_auc)


def get_metrics(model_in, test_dl, thresh=0.5, param_count=False):
    
    if (param_count):
        
        total_params = sum(p.numel() for p in model_in.parameters())
        print("Total number of parameters is: ", total_params)
        
        total_trainable_params = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
        print("Total number of trainable parameters is: ", total_trainable_params)
        
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