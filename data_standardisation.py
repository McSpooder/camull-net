'''The following script contains functions for normalising and standardising the input data. Although the training data has already been standardised,
the user inputted data is not.'''

from sklearn import preprocessing
import numpy as np

min_max = {"AGE":(54.4, 91.4), "PTEDUCAT":(4, 20), "CDRSB":(0,17), "ADAS11":(0,59), "ADAS13":(0,74), "RAVLT_immediate":(1,75), "RAVLT_learning":(-5,14), "RAVLT_forgetting":(-1035,15),
            "RAVLT_perc_forgetting":(-9409,100)}

encodings = {"GENDER":{"Female":0, "Male":1}, "RACCAT":{"Am Indian/Alaskan": 0, "Asian": 1, "Black": 2, "Haiwaiian/Other PI": 3, "More than one": 4, "Unknown": 5, "White":6}}

column_names_to_normalize = ["AGE", "PTEDUCAT", "CDRSB", "ADAS11", "ADAS13", "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting"]
column_names_to_encode = ["GENDER", "ETHCAT", "RACCAT"]


def min_max_scale(in_dict):
    #here is some code for min maxing stuff
    for key in in_dict.keys():
        if key in column_names_to_normalize:
            X = in_dict[key]
            X_std = ((X - min_max[key][0]) / (min_max[key][1] - min_max[key][0]))
            X_scaled = X_std * (min_max[key][1] - min_max[key][0]) + min_max[key][0]
            in_dict[key] = X_scaled

    return in_dict

def to_categorical(key, value):
    return encodings[key][value]


def one_hot_encode(in_dict, clinical):

    
    if (in_dict["ETHNICITY"] == "Hisp/Latino"):
        clinical[11] = 1
    elif (in_dict["ETHNICITY"] == "Not Hisp/Latino"):
        clinical[12] = 1
    else: #unknown
        clinical[13] = 1


    if (in_dict["RACCAT"] == "Am Indian/Alaskan"):
        clinical[14] = 1
    elif (in_dict["RACCAT"] == "Asian"):
        clinical[15] = 1
    elif (in_dict["RACCAT"] == "Black"):
        clinical[16] = 1
    elif (in_dict["RACCAT"] == "Haiwaiian/Other PI"):
        clinical[17] = 1
    elif (in_dict["RACCAT"] == "More than one"):
        clinical[18] = 1
    elif (in_dict["RACCAT"] == "Unknown"):
        clinical[19] = 1
    elif (in_dict["RACCAT"] == "White"):
        clinical[20] = 1

    return clinical

def convert_to_np(in_dict):
    '''Gets clinical features vector from unormalized user inputted values. This would be used during inference to change user inputted values.'''
    #to do: normalize these assignments
    clinical = np.zeros(21)
    in_dict = min_max_scale(in_dict)

    clinical[0] = in_dict["AGE"] 
    clinical[1] = in_dict["PTEDUCAT"]
    clinical[2] = in_dict["APOE4"]
    clinical[3] = in_dict["CDRSB"]
    clinical[4] = in_dict["ADAS11"]
    clinical[5] = in_dict["ADAS13"]
    clinical[6] = in_dict["RAVLT_immediate"]
    clinical[7] = in_dict["RAVLT_learning"]
    clinical[8] = in_dict["RAVLT_forgetting"]
    clinical[9] = in_dict["RAVLT_perc_forgetting"]
    clinical[10] = encodings["GENDER"][in_dict["GENDER"]]
    
    clinical = one_hot_encode(in_dict, clinical)

    return clinical
    
