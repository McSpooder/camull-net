'''The following module trains the weights of the neural network model.'''
import os
import datetime
import uuid
from tqdm.auto import tqdm

import torch
import torch.nn    as nn
import torch.optim as optim

from data_declaration import Task
from loader_helper    import LoaderHelper
from architecture     import load_cam_model, Camull
from evaluation import evaluate_model

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

def save_weights(model_in, uuid_arg, fold=1, task: Task = None):
    '''The following function saves the weights file into required folder'''
    root_path = ""

    if task == Task.CN_v_AD:
        root_path = "../weights/CN_v_AD/"     + uuid_arg + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + uuid_arg + "/"

    if fold == 1:
        os.mkdir(root_path) #otherwise it already exists

    while True:

        s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format(fold, date=datetime.datetime.now()) # pylint: disable=line-too-long

        if os.path.exists(s_path):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break

def load_model(arch, path=None):
    '''Function for loaded camull net from a specified weights path'''
    if arch == "camull": #must be camull

        if path is None:
            model = load_cam_model("../weights/camnet/fold_0_weights-2020-04-09_18_29_02")
        else:
            model = load_cam_model(path)


    return model

def build_arch():
    '''Function for instantiating the pytorch neural network object'''
    net = Camull()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(DEVICE)
    net.double()

    return net


def train_loop(model_in, train_dl, epochs):
    '''Function containing the neural net model training loop'''
    optimizer = optim.Adam(model_in.parameters(), lr=0.001, weight_decay=5e-5)

    loss_function = nn.BCELoss()

    model_in.train()

    for i in range(epochs):

        for _, sample_batched in enumerate(tqdm(train_dl)):

            batch_x = sample_batched['mri'].to(DEVICE)
            batch_xb = sample_batched['clin_t'].to(DEVICE)
            batch_y = sample_batched['label'].to(DEVICE)

            model_in.zero_grad()
            outputs = model_in((batch_x, batch_xb))

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss.item(), 5)))


def train_camull(ld_helper, k_folds=5, model=None, epochs=40):
    '''The function for training the camull network'''
    task = ld_helper.get_task()
    uuid_ = uuid.uuid4().hex
    model_cop = model

    for k_ind in range(k_folds):

        if model_cop is None:
            model = build_arch()
        else:
            model = model_cop

        train_dl = ld_helper.get_train_dl(k_ind)
        train_loop(model, train_dl, epochs)
        save_weights(model, uuid_, fold=k_ind+1, task=task)

        print("Completed fold {}/{}.".format(k_ind, k_folds))

    return uuid_

def start(ld_helper, epochs=40, model_uuid=None):

    task = ld_helper.get_task()
    if (task == Task.CN_v_AD):
        model_uuid = train_camull(ld_helper, epochs=40)
    else: # sMCI vs pMCI
        if (model_uuid != None):
            model = load_model("camull", model_uuid)
            model_uuid  = train_camull(ld_helper, model=model, epochs=40)
        else:
            print("Need a model uuid")
    
    return model_uuid


def main():
    '''Main function of the module.'''
    #CN v AD
    #ld_helper = LoaderHelper(task=Task.CN_v_AD)
    #model_uuid = train_camull(ld_helper, epochs=40)
    #evaluate_model(DEVICE, "c51bf83c4455416e8bc8b1ebbc8b75ca", ld_helper)

    #transfer learning for pMCI v sMCI
    # ld_helper.change_task(Task.sMCI_v_pMCI)
    # model = load_model("camull", uuid)
    # uuid  = train_camull(ld_helper, model=model, epochs=40)
    # evaluate_model(device, uuid, ld_helper)

main()
