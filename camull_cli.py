from train_model          import start
from evaluation           import evaluate_fold, evaluate_model
from data_declaration     import Task
from loader_helper        import LoaderHelper
from architecture         import load_cam_model
from data_declaration     import get_mri
from data_standardisation import convert_to_np

import torch

import sqlite3
import os
import glob

global conn
global cur

if not (os.path.exists("../weights")):
    os.mkdir("../weights")
    conn = sqlite3.connect("../weights/neural-network.db")
    cur = conn.cursor()
    sql_create_projects_table = """ CREATE TABLE nn_perfomance (
                                        model_uuid integer PRIMARY KEY NOT NULL,
                                        time datetime,
                                        model_task text,
                                        accuracy double
                                        sensitivity double,
                                        specificity double,
                                        roc_auc double
                                    ); """
    cur.execute(sql_create_projects_table)
else:
    conn = sqlite3.connect("../weights/neural-network.db")
    cur = conn.cursor()

    

def advanced_run():
    '''The advanced run allows the user to tweak the hyper-parameters.'''

def basic_run(device):
    '''The basic run doesn't give the user option to tweak the hyper-parameters.'''

    print("\n")
    print("Welcome to Camull.\n")
    print("         |")
    print("    m1a  |")
    print("         |")
    print("     /   |   \\    ")
    print("     \   |   /    1. Train a new model.")
    print("   .  --\|/--  ,  2. Train an existing model or do transfer learning.")
    print("    '--|___|--'   3. Make an inference with an existing model.")
    print("    ,--|___|--,   4. Evaluate a model. ")
    print("   '  /\o o/\  `    ")
    print("     +   +   +    ")
    print("      `     '    ")
    print("\n")

    choice = input("Please input a choice from the menu above?: ")
    print("\n")

    if (int(choice) == 1):
        train_new_model_cli(device)    
    elif (int(choice) == 2):
        transfer_learning(device)
    elif (int(choice) == 3):
        make_an_inference(device)

def make_an_inference(device):
    print("\n")
    print("0. NC vs AD")
    print("1. sMCI vs pMCI")
    print("\n")
    choice = int(input("Which task would you like to perform?: "))

    if (choice == 0):
        #fetch the most recent models for NC vs AD.
        print("\n")
        print("Here are the 5 most recent models trained for NC vs AD.")
        print("model uuid | Time | model task | accuracy | sensitivity | specificity | roc_auc")
        result = cur.execute("SELECT * FROM nn_perfomance WHERE task is 'NC_v_AD' ORDER BY time DESC LIMIT 5")
        model_uuids = []
        for i, row in enumerate(result):
            print(i+1, row)
            model_uuids.append(row[0])
        print("\n")
        choice = int(input("Please enter an index to use [1, 5]: "))
        path_a = "../weights/{}/{}/fold_1_*".format(str(Task(1)), model_uuids[i])
        path = glob.glob(path_a)[0]
        if (int(choice) != 6):
            mri, clinical = get_subject_info()
            mri_t = torch.from_numpy(mri) / 255.0 
            mri_t = mri_t.unsqueeze(0)
            mri_t = mri_t.to(device)
            clin_t = torch.from_numpy(clinical)
            clin_t = clin_t.unsqueeze(0)
            clin_t = clin_t.to(device)
            model = load_cam_model(path)
            model.eval()

            net_out = -1
            with torch.no_grad():
                net_out = model((mri_t.view(-1, 1, 110, 110, 110), clin_t.view(1, 21)))
                print("The probability is: ", net_out)
        else:
            print("Invalid selection.")

    else:
        #fetch the most recent models for sMCI vs pMCI
        pass

def transfer_learning(device):
    print("To train for sMCI vs pMCI you need transfer learning from a NC vs AD model. Would you like to transfer learning from an existing model or train a new NC vs AD model?\n")
    print("0. Existing model.")
    print("1. Train a new NC vs AD model.\n")
    choice = input("Please select an option: ")
    if (int(choice) == 0):

        print("Here are 10 of your most recent NC v AD models.")
        print("model uuid | Time | model task | accuracy | sensitivity | specificity | roc_auc")
        model_uuids = fetch_models_from_db()

        choice = input("Please enter the model number [1, 10] or the uuid that you would like to choose:")
        
        
        if (int(choice) != 5):
            ld_helper = LoaderHelper(Task.sMCI_v_pMCI)
            uuid = start(ld_helper, 40, model_uuids[int(choice)])
            print("\n")
            print("A new sMCI vs pMCI model has been trained under the tag: {}".format(uuid))
            choice = input("Would you like to evaluate it (Y/n)?")
            print("\n")
            if (int(choice) == 'y' or 'Y' or ''):
                evaluate_model(device, uuid, ld_helper)
            else:
                print("Please enter a uuid.")

    else:
        print("Training a new NC vs AD model.")
        print("\n")
        uuid = train_new_model_cli()
        ld_helper = LoaderHelper(Task.sMCI_v_pMCI)
        choice = input("How many epochs would you like to train the task sMCI vs pMCI?(default:40): ")
        if choice != "":
            valid = False
            while(valid==False):
                try:
                    uuid = start(ld_helper, int(choice), uuid)
                    valid = True
                except:
                    print("Please input a valid number.")
        else:
            print("Training a new model.")
            uuid = start(ld_helper, 40, uuid)

def train_new_model_cli(device):

    task = Task.NC_v_AD
    ld_helper = LoaderHelper(task)

    if (True):
        print("\n")
        epochs = input("How many epochs would you like to do? (default: 40): ")
        print("\n")

        uuid = ""
        if epochs == "":
            uuid = start(device, ld_helper, 40)
        else:
            num_epochs = 0
            try:
                num_epochs = int(epochs)
            except:
                print("Number of epochs must be a valid number.")
            uuid = start(device, ld_helper, int(epochs))

        print("\n")
        print("A new NC vs AD model has been trained under the tag: {}".format(uuid))
        print("\n")
        print("Would you like to evaluate it?")
        print("0. Yes")
        print("1. No")
        print("\n")
        choice = input("Enter your choice [0,1]: ")
        if (int(choice) == 0):
            print("\n")
            print("There are 5 folds to evaluate")
            print("Input a fold number to evaluate or input 6 to evaluate all folds.")
            print("\n")
            choice = int(input("Enter your choice [1,6]: "))
            if (choice != 6):
                evaluate_fold(device, uuid, ld_helper, choice)
            else:
                evaluate_model(device, uuid, ld_helper)
        else:
            basic_run(device)
    else:
        print("To train for sMCI vs pMCI you need transfer learning from a NC vs AD model. Would you like to transfer learning from an existing model or train a new NC vs AD model?\n")
        print("0. Existing model.")
        print("1. Train a new NC vs AD model.\n")
        choice = input("Please select an option: ")


def fetch_models_from_db():
    global conn
    global cur
    model_uuids = []
    i = 0
    for i, row in enumerate(cur.execute('SELECT * FROM nn_perfomance')):
        print(row)
        model_uuids[i] = row(i)
    return model_uuids


def get_subject_infor_from_db():
    global conn
    global cur
    subjects = []
    i = 0
    for i, row in enumerate(cur.execute('SELECT * FROM patient_clinical')):
        print(row)
        subjects[i] = row
    return subjects

def get_subject_info():

    path = input("Please provide the path to the MRI scan: ")
    mri = get_mri(path)

    clinical = {}

    clinical["AGE"] = float(input("What is the age of the subject?: "))
    clinical["GENDER"] = input("What is the subjects gender?: ")
    clinical["ETHNICITY"] = input("What is the subjects ethnicity?: ")
    clinical["RACCAT"] = input("What is the subjects race?: ")
    clinical["PTEDUCAT"] = float(input("How many years did the patient spent in education?: "))
    clinical["APOE4"] = float(input("How many apoe4 genes does the subject have?: "))
    clinical["CDRSB"] = float(input("What is the subjects CDRSB score?: "))
    clinical["ADAS11"] = float(input("What is the subjects ADAS11 score?: "))
    clinical["ADAS13"] = float(input("What is the subjects ADAS13 score?: "))
    clinical["RAVLT_immediate"] = float(input("What is the subjects RAVLT_immediate score?: "))
    clinical["RAVLT_learning"] = float(input("What is the subjects RAVLT_learning score?: "))
    clinical["RAVLT_forgetting"] = float(input("What is the subjects RAVLT_forgetting score?: "))
    clinical["RAVLT_perc_forgetting"] = float(input("What is the subjects RAVLT percentage forgetting score?"))
    
    clinical = convert_to_np(clinical)
    return (mri, clinical)


def __main__():

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    basic_run(device)


if __name__ == '__main__':
    __main__()