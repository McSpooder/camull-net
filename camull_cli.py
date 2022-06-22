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
sql_create_basic_table = """ CREATE TABLE IF NOT EXISTS nn_basic (
                            model_id INTEGER PRIMARY KEY NOT NULL,
                            model_uuid text,
                            time datetime,
                            model_task text
                        ); """
sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS nn_perfomance (
                                    eval_id INTEGER PRIMARY KEY NOT NULL,
                                    model_uuid text,
                                    time datetime,
                                    model_task text,
                                    accuracy double,
                                    sensitivity double,
                                    specificity double,
                                    roc_auc double
                                ); """
cur.execute(sql_create_basic_table)
cur.execute(sql_create_projects_table)

def check_unevaluated():
    unevaluated_count = """ SELECT model_uuid 
                            FROM   nn_basic 
                            WHERE  model_uuid NOT IN (SELECT model_uuid FROM nn_perfomance)"""
    out = cur.execute(unevaluated_count)
    for i, row in enumerate(out):
        continue
    print("You have {} models that can be evaluated.".format(i))


def advanced_run():
    '''The advanced run allows the user to tweak the hyper-parameters.'''

def basic_run(device):
    '''The basic run doesn't give the user option to tweak the hyper-parameters.'''
    check_unevaluated()

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
    print("   '  /\o o/\  `  5. Display the most recent evaluated models.")
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
    elif (int(choice) == 4):
        evaluate_a_model(device)
    elif (int(choice) == 5):
        fetch_models_from_db(evaluated=True)
        basic_run(device)

def make_an_inference(device, mri=None, clin=None):

    print("\n")
    print("1. NC vs AD")
    print("2. sMCI vs pMCI")
    print("\n")
    choice = int(input("Which task would you like to perform?: "))
    
    if choice == 1: task = Task.NC_v_AD
    else: task = Task.sMCI_v_pMCI

    #fetch the most recent models for NC vs AD.
    print("\n")
    print("Here are the 5 most recent models trained for {}.".format(str(task)))
    print("    model uuid               | Time      | model task | accuracy | sensitivity | specificity | roc_auc")
    result = cur.execute("SELECT * FROM nn_perfomance WHERE model_task is '{}' ORDER BY time DESC LIMIT 5".format(str(task)))
    model_uuids = []
    for i, row in enumerate(result):
        print(i+1, row)
        model_uuids.append(row[1])
    print("\n")
    choice = int(input("Please enter an index to use [1, 5]: "))
    path_a = "../weights/{}/{}/*".format(str(task), model_uuids[choice-1])
    path = glob.glob(path_a)[0]
    while (True):
        if (int(choice) != 6):
            if torch.is_tensor(mri) == False:
                mri, clinical = get_subject_info()
                #Performs necessery transformations for the neural network
                mri = torch.from_numpy(mri) / 255.0 
                mri = mri.unsqueeze(0)
                mri = mri.to(device)
                clin = torch.from_numpy(clinical)
                clin = clin.unsqueeze(0)
                clin = clin.to(device)

            model = load_cam_model(path)
            model.eval()

            net_out = -1
            with torch.no_grad():
                net_out = model((mri.view(-1, 1, 110, 110, 110), clin.view(1, 21)))
                if task == Task.NC_v_AD:
                    print("The probability that the subject has AD is " + str(net_out[0].item()*100)  + "%")
                else:
                    print("The probability that the subject will convert to AD within 3 years is " + str(net_out[0].item()*100)  + "%")
                print("\n")
                print("The probability is deterministic and hinges upon the neural network model and subject data.")
                print("\n")
                print("1. Input different subject details for the same model.")
                print("2. Use the same subject details for inference with another model.")
                print("3. Make a new inference with another model.")
                print("4. Return to the main menu.")
                print("\n")
                choice = int(input("Please enter your choice: "))
                if choice == 4:
                    basic_run(device)
                elif choice == 3:
                    make_an_inference(device)
                elif choice == 2:
                    make_an_inference(device, mri, clin)
                elif choice == 1:
                    mri = None
                    clin = None
                    continue
   

def transfer_learning(device):
    print("To train for sMCI vs pMCI you need transfer learning from a NC vs AD model. Would you like to transfer learning from an existing model or train a new NC vs AD model?\n")
    print("1. Existing model.")
    print("2. Train a new NC vs AD model.\n")
    choice = input("Please select an option: ")
    if (int(choice) == 1):

        print("Here are 10 of your most recent NC v AD models.")
        print("    model uuid               | Time      | model task | accuracy | sensitivity | specificity | roc_auc")
        model_uuids = fetch_models_from_db(task=Task.NC_v_AD)

        if not model_uuids == []:
            choice = input("Please enter the model number [1, 10] or the uuid that you would like to choose:")
            model_uuid = model_uuids[int(choice)-1]
            ld_helper = LoaderHelper(Task.sMCI_v_pMCI)
            choice = input("How many epochs would you like to train the task sMCI vs pMCI?(default:40): ")
            uuid = start(device, ld_helper, int(choice), model_uuid)
            print("\n")
            print("\n")
            print("A new sMCI v pMCI model has been trained under the tag: {}".format(uuid))
            print("\n")
            print("Would you like to evaluate it? You must do so for it to be saved to the database.")
            print("\n")
            choice = input("Enter your choice [Y/n]: ")
            if choice == 'Y' or 'y' or '': 
                choice = 1 
            else: 
                choice = 0
            if (int(choice) == 1):
                print("\n")
                print("There are 5 folds to evaluate")
                print("Input a fold number to evaluate or input 6 to evaluate all folds.")
                print("\n")
                choice = int(input("Enter your choice [1,6]: "))
                if (choice != 6):
                    evaluate_fold(device, uuid, ld_helper, choice, cur)
                else:
                    evaluate_model(device, uuid, ld_helper, cur)

        else:
            print("\n")
            print("No models available. Please train a new model.")
            choice = input("Would you like to train a new model[Y/n]?: ")
            if choice  == 'y' or 'Y' or '': choice = 2

    if (int(choice) == 2):
        print("Training a new NC vs AD model.")
        print("\n")
        uuid = train_new_model_cli(device)
        ld_helper = LoaderHelper(Task.sMCI_v_pMCI)
        choice = input("How many epochs would you like to train the task sMCI vs pMCI?(default:40): ")
        print("\n")
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
    print("The model has been trained.")
    basic_run(device)

def train_new_model_cli(device):

    print("\n")
    print("Camull-Net can be trained for two seperate tasks.")
    print("1. NC v AD : Distinguishing between subjects from the normal cohort group and subjects with Alzheimers Disease.")
    print("2. sMCI v pMCI : Distinguishing between subjects with static mild cognitive impairement and progressive mild cognitive impairement.")
    print("\n")
    choice = input("Please enter your choice [(1),2]:")
    if choice == "" : choice = 1
    else: choice = int(choice)

    if choice == 1: task = Task.NC_v_AD
    else: task = Task.sMCI_v_pMCI

    ld_helper = LoaderHelper(task)

    print("\n")
    epochs = input("How many epochs would you like to do? (default: 40): ")
    print("\n")

    uuid = ""
    if epochs == "":
        uuid = start(device, ld_helper, 40)
    else:
        try:
            uuid = start(device, ld_helper, int(epochs))
        except:
            print("Number of epochs must be a valid number.")

    print("\n")
    print("A new {} model has been trained under the tag: {}".format(str(task), uuid))
    print("\n")
    print("Would you like to evaluate it? You must do so for it to be saved to the database.")
    print("\n")
    choice = input("Enter your choice [Y/n]: ")
    if choice == 'Y' or 'y' or '': 
        choice = 1 
    else: 
        choice = 0
    if (int(choice) == 1):
        print("\n")
        print("There are 5 folds to evaluate")
        print("Input a fold number to evaluate or input 6 to evaluate all folds.")
        print("\n")
        choice = int(input("Enter your choice [1,6]: "))
        if (choice != 6):
            evaluate_fold(device, uuid, ld_helper, choice, cur)
        else:
            evaluate_model(device, uuid, ld_helper, cur)
    else:
        basic_run(device) #loop back round to the start of the program.


def evaluate_a_model(device):
    print("Which task would you like to evaluate?")
    print("1) NC v AD")
    print("2) sMCI v pMCI")

    choice = int(input("Please enter your input [1..2]: "))
    task = Task
    if choice == 1:
        task = Task.NC_v_AD
    else:
        task = Task.sMCI_v_pMCI

    print("Here are 10 of your most recent unevaluated {} models.".format(str(task)))
    print("\n")
    print("    id  | model uuid               | Time      | model task")
    model_uuids = fetch_models_from_db(task)
    target_uuid = ""

    if not model_uuids == []:
        valid = False
        while (not valid):
            choice = input("Please enter the model number [1, 10] or the uuid that you would like to choose:")
            try:
                target_uuid = model_uuids[int(choice)-1]
                valid = True
            except:
                for i in range(len(model_uuids)):
                    if choice == model_uuids:
                        target_uuid = choice
                        break
                print("Please enter a valid input.")
        
        ld_helper = LoaderHelper(task)
        print("There are 5 folds to evaluate")
        print("Input a fold number to evaluate or input 6 to evaluate all folds.")
        print("\n")
        choice = int(input("Enter your choice [1,6]: "))
        if (choice != 6):
            evaluate_fold(device, target_uuid, ld_helper, choice, commit_to_db=True)
            remove_from_db(target_uuid)
        else:
            evaluate_model(device, target_uuid, ld_helper, cur)
            remove_from_db(target_uuid)

        print("The model has been evaluated. Check the graphs folder and look at the metrics in the database.")
        basic_run(device)

    else:
        print("\n")
        print("No models available. Please train a new model.")
        choice = input("Would you like to train a new model[Y/n]?: ")
        if choice  == 'y' or 'Y' or '': train_new_model_cli 
        else: basic_run

def fetch_models_from_db(task=None, evaluated=False):
    global conn
    global cur
    model_uuids = []

    i = 0
    if task == None: sql_querry = "SELECT * FROM nn_perfomance"
    else: sql_querry = "SELECT * FROM nn_basic WHERE model_task is '" + str(task) + "'"

    for i, row in enumerate(cur.execute(sql_querry)):
        print(str(i+1) + ". ", row)
        model_uuids.append(row[1])
    return model_uuids


def remove_from_db(model_uuid):
    sql_statement = "DELETE FROM nn_basic WHERE model_uuid is '" + model_uuid + "';"
    cur.execute(sql_statement)


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

    path = input("Please provide the path to the MRI scan (DEFAULT: ./inference-sample/ad-scan.nii): ")
    if path == "":
        path = "./inference-sample/ad-scan.nii"
    mri = get_mri(path)

    clinical = {}

    clinical["AGE"] = float(input("What is the age of the subject?[0..100]: "))
    print("\n")
    print("a) Female")
    print("b) Male")
    print("\n")
    clinical["GENDER"] = input("What is the subjects gender?(write it out fully):")
    print("\n")
    print("a) Not Hisp/Latino")
    print("b) Hisp/Latino")
    print("\n")
    clinical["ETHNICITY"] = input("What is the subjects ethnicity?(Write it out fully): ")
    print("\n")
    print("Here are the available racial categories. Please choose one from the above: ")
    print("\n")
    print("a) Am Indian/Alaskan")
    print("b) Asian")
    print("c) Black")
    print("d) Haiwaiian/Other PI")
    print("e) More than one")
    print("f) Unknown")
    print("g) White")
    print("\n")
    clinical["RACCAT"] = input("What is the subjects race?(Write it out fully): ")
    clinical["PTEDUCAT"] = float(input("How many years did the patient spent in education?[4..20]: "))
    clinical["APOE4"] = float(input("How many apoe4 genes does the subject have?[1..2]: "))
    clinical["CDRSB"] = float(input("What is the subjects CDRSB score?[0..17]: "))
    clinical["ADAS11"] = float(input("What is the subjects ADAS11 score?[0..59]: "))
    clinical["ADAS13"] = float(input("What is the subjects ADAS13 score?[0..74]: "))
    clinical["RAVLT_immediate"] = float(input("What is the subjects RAVLT_immediate score?[0..75]: "))
    clinical["RAVLT_learning"] = float(input("What is the subjects RAVLT_learning score?[0..14]: "))
    clinical["RAVLT_forgetting"] = float(input("What is the subjects RAVLT_forgetting score?[0,15]: "))
    clinical["RAVLT_perc_forgetting"] = float(input("What is the subjects RAVLT percentage forgetting score?[0..100]:"))
    print("\n")
    
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