'''The following module trains the weights of the neural network model.'''

import glob
from data_declaration import Task
from architecture     import load_cam_model, Camull

import torch
import torch.nn    as nn
import torch.optim as optim

import enlighten
from tqdm.auto import tqdm

import os
import sys
import datetime
import uuid

DEVICE = None

def start(device, ld_helper, epochs, model_uuid=None):
        
    def load_model(arch, model_uuid=None):
        '''Function for loaded camull net from a specified weights path'''
        if arch == "camull": #must be camull

            if model_uuid is None:
                model = load_cam_model("../weights/camnet/fold_0_weights-2020-04-09_18_29_02")
            else:
                paths = glob.glob("../weights/NC_v_AD/{}/*".format(model_uuid))              
                model = load_cam_model(paths[0])

        return model

    def save_weights(model_in, uuid_arg, fold=1, task: Task = None):
        '''The following function saves the weights file into required folder'''

        if sys.platform.__str__() == 'linux':
            root_path = "../weights/" + task.__str__() + "/"     + uuid_arg + "/"
        else: #windows
            root_path = "..\\weights\\" + task.__str__() + "\\"     + uuid_arg + "\\"


        if fold == 1: os.makedirs(root_path, exist_ok=True)


        while True:

            s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H-%M-%S}.pt".format(fold, date=datetime.datetime.now()) # pylint: disable=line-too-long

            if os.path.exists(s_path):
                print("Path exists. Choosing another path.")
            else:
                #check if the directory exists
                torch.save(model_in, s_path)
                break


    def build_arch():
        '''Function for instantiating the pytorch neural network object'''
        net = Camull()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        net.to(DEVICE)
        net.double()

        return net

    def train_camull(ld_helper, k_folds=5, model=None, epochs=40):
        '''The function for training the camull network'''

        folds_c.total = k_folds

        task = ld_helper.get_task()
        uuid_ = uuid.uuid4().hex
        model_cop = model

        for k_ind in range(k_folds):

            print("\n")
            print("=========== Training on Fold {} ===========".format(k_ind))

            if model_cop is None:
                model = build_arch()
            else:
                model = model_cop

            train_dl = ld_helper.get_train_dl(k_ind)
            train_loop(model, train_dl, epochs)
            save_weights(model, uuid_, fold=k_ind+1, task=task)

            folds_c.update()


        folds_c.count = 0

        return uuid_

    def train_loop(model_in, train_dl, epochs):
        '''Function containing the neural net model training loop'''

        epochs_c.total = epochs
        batches_c.total = len(train_dl)

        optimizer = optim.Adam(model_in.parameters(), lr=0.001, weight_decay=5e-5)

        loss_function = nn.BCELoss()

        model_in.train()

        for i in range(epochs):

            for _, sample_batched in enumerate(train_dl):

                batch_x = sample_batched['mri'].to(DEVICE)
                batch_xb = sample_batched['clin_t'].to(DEVICE)
                batch_y = sample_batched['label'].to(DEVICE)

                model_in.zero_grad()
                outputs = model_in((batch_x, batch_xb))

                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()
                batches_c.update()
            
            batches_c.count = 0

            print("Epoch: {}/{}, train loss: {}".format(i+1, epochs, round(loss.item(), 5)))
            epochs_c.update()

        epochs_c.count = 0
        #epochs_c.update()


    manager = enlighten.get_manager()
    folds_c = manager.counter(total=5, desc='Fold', unit='folds')
    epochs_c = manager.counter(total=epochs, desc='Epochs', unit='epochs')
    batches_c = manager.counter(total=75, desc='Batches', unit='batches')

    task = ld_helper.get_task()
    DEVICE = device
    if (task == Task.NC_v_AD):
        model_uuid = train_camull(ld_helper=ld_helper, epochs=epochs)
    else: # sMCI vs pMCI
        if (model_uuid != None):
            model = load_model("camull", model_uuid)
            model_uuid  = train_camull(ld_helper, model=model, epochs=epochs)
        else:
            print("Need a model uuid")
    
    return model_uuid


