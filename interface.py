from train_model import start
from evaluation import evaluate_model
from data_declaration import Task
from loader_helper    import LoaderHelper
from architecture import load_cam_model
from data_declaration import get_mri
from data_standardisation import convert_to_np

import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def train_new_model_cli():

    print("0. NC vs AD")
    print("1. sMCI vs pMCI")
    print("\n")
    choice = input("Which task would you like to perform?: ")
    print("\n")

    task = Task.NC_v_AD
    ld_helper = LoaderHelper(task)

    if (int(choice) == 0):
        uuid = start(ld_helper, 40)
        print("A new NC vs AD model has been trained under the tag: {}".format(uuid))
        print("Would you like to evaluate it?")
        print("0. Yes")
        print("1. No")
        if (int(choice) == 0):
            evaluate_model(device, uuid, ld_helper)
    else:
        print("To train for sMCI vs pMCI you need transfer learning from a NC vs AD model. Would you like to transfer learning from an existing model or train a new NC vs AD model?\n")
        print("0. Existing model.")
        print("1. Train a new NC vs AD model.\n")
        choice = input("Please select an option: ")
        if (int(choice) == 0):
            #Display the 5 most recent models.
            model_uuids = ["as3asf34f352f43t42w90asf3e", "86ft3nfa9nf302yns273nds82n", "872b17sd271dn27717rsnsaf31", "86ft3nfa9nf302yns273nds82n", "86ft3nfa9nf302yns273nds82n"]
            print("Here are the 5 most recent models...(dummies)")
            print("0. {} 01/02/20T12:02:31:03 avg. auc={}".format(model_uuids[0], 0))
            print("1. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[1], 0))
            print("2. {} 01/02/20T12:06:20:01 avg. auc={}".format(model_uuids[2], 0))
            print("3. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[3], 0))
            print("4. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[4], 0))
            print("5. Other.")
            print("\n")
            choice = input("Please select an option: ")
            if (int(choice) != 5):
                ld_helper.change_task(Task.sMCI_v_pMCI)
                uuid = start(ld_helper, 40, model_uuids[int(choice)])
                print("A new NC vs AD model has been trained under the tag: {}".format(uuid))
                print("Would you like to evaluate it?")
                print("0. Yes")
                print("1. No")
                if (int(choice) == 0):
                    evaluate_model(device, uuid, ld_helper)
                else:
                    print("Please enter a uuid.")
        else:
            print("Training a new NC vs AD model.")
            uuid = start(ld_helper, 40)
            print("A new NC vs AD model has been generated under the tag: {}.".format(uuid))
            print("Initiating transfer learning for the sMCI vs pMCI task.")
            ld_helper.change_task(Task.sMCI_v_pMCI)
            uuid = start(ld_helper, 40, uuid)
            print("A new sMCI vs pMCI model under the tag {} has been trained.\n".format(uuid))
            print("Would you like to evaluate the model?")
            print("0. Yes")
            print("1. No")
            choice = input("Please select an option: ")
            if (int(choice) == 0):
                evaluate_model(device, uuid, ld_helper)

def train_an_existing_model():
    print("yo.")

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


def make_an_inference():
    print("0. NC vs AD")
    print("1. sMCI vs pMCI")
    print("\n")
    choice = int(input("Which task would you like to perform?: "))

    if (choice == 0):
        #fetch the most recent models for NC vs AD.
        model_uuids = ["as3asf34f352f43t42w90asf3e", "86ft3nfa9nf302yns273nds82n", "872b17sd271dn27717rsnsaf31", "86ft3nfa9nf302yns273nds82n", "86ft3nfa9nf302yns273nds82n"]
        print("Here are the 5 most recent models trained for NC vs AD...(dummies)")
        print("0. {} 01/02/20T12:02:31:03 avg. auc={}".format(model_uuids[0], 0))
        print("1. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[1], 0))
        print("2. {} 01/02/20T12:06:20:01 avg. auc={}".format(model_uuids[2], 0))
        print("3. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[3], 0))
        print("4. {} 01/02/20T12:03:31:03 avg. auc={}".format(model_uuids[4], 0))
        print("5. Other.")
        print("\n")
        choice = input("Please select an option: ")
        path = "/home/dan/Programming/weights/NC_v_AD/c51bf83c4455416e8bc8b1ebbc8b75ca/fold_2_weights-2020-04-29_13_04_53"
        if (int(choice) != 5):
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

def basic_run():
    '''The basic run doesn't give the user option to tweak the hyper-parameters.'''

    print("Welcome to Camull.\n")
    print("0. Train a new model.")
    print("1. Test an existing model.")
    print("2. Make an inference with an existing model.")
    print("\n")

    choice = input("Please input a choice from the menu above?: ")
    print("\n")

    if (int(choice) == 0):
        train_new_model_cli()    
    elif (int(choice) == 1):
        train_an_existing_model()
    elif (int(choice) == 2):
        make_an_inference()

def advanced_run():
    '''The advanced run allows the user to tweak the hyper-parameters.'''

def __main__():
    basic_run()

__main__()