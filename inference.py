from pathlib import Path
from data_declaration import DataSample, Task, ToTensor
from loader_helper    import loader_helper
from architecture import load_cam_model
from torchvision import transforms
import torch

def __main__(str):
    print("alternative baby")

def __main__():
    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if (Path('../weights').exists()):
        #task = input("Please specify what task you want to perform [1:\'CN vs AD\', 2:\'sMCI vs pMCI\']: ")
        task = 0
        ld_helper = None

        if (task == 0):
            ld_helper = loader_helper(task=Task.CN_v_AD)
        else:
            ld_helper = loader_helper(task=Task.sMCI_v_pMCI)

        #weights_path = input("Please specify the directory of the weights file that you want to use: ") # should be relative to the proj folder and use forward slashes.
        weights_path = "../weights/CN_v_AD/c51bf83c4455416e8bc8b1ebbc8b75ca/fold_5_weights-2020-04-29_13_17_33"
        print("\n")
        print("Loading camull for inference...")
        model = load_cam_model(weights_path)
        print("Success.")

        try:

            #sample_path = input("Please specify the path of the data sample folder (contains .nii and csv record): ")
            sample_path = "../data/inference"

            datasample = DataSample(sample_path,transform=transforms.Compose([
                        ToTensor()
                    ]))


            mri = datasample['mri'].to(device)
            clinical = datasample['clin_t'].to(device)

            model.eval()

            with torch.no_grad():
                outputs = model((mri.view(-1, 1, 110, 110, 110), clinical.view(1, 21)))
                
        except Exception as e:
            print(e)
    else:
        print("You must either place a weights file in the weights folder or train the model by running: python train_model.py")


def __cycle__():
    print("blah")

__main__()