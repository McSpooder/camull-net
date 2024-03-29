# camull-net

![](https://i.gyazo.com/0415de280214e761763451cc470af498.png)

An implementation of the Cambridge parameter efficient 3D convolutional network for the classification of Alzheimer's disease (AD) in pytorch. The architecture is described in detail in this [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191930031X "paper"). More specifically the architecture can classify between static and progressive Mild cognitive impairement (sMCI vs pMCI) which is a disease that precludes Alzheimers. For this latter task the predictions are for the subsequent 3 years after the subject has been analysed. The area under the curve achieved for this harder second task is: 0.88.

The code was developed as part of my dissertation at the university of Hull. You can read more about it, including the data required, in this medium [link](https://medium.com/@hextra_19712/deep-learning-for-alzheimers-classification-57611161e442 "article").

## Features

* Program
  * Comes with pre-trained models for alzheimers prediction and mild cognitive impairment conversion.
  * Performs k-fold cross validation
  * Loads and saves neural network models
  * Logs the train and test loss
  * Outputs Area under the ROC curve jpegs for each k-fold
  * Saves each model under a unique identifier in a neat directory structure
  * Comes with an interface for submitting clinical user data.
* Neural Net
  * Distinguishes between subjects that are normal and that have alzeimers disease.
  * Can predict whether a subject will convert to alzheimers disease within 3 years.
  * 3D seperable convolution to reduce number of parameters and hence overfitting.
  * Multiple concurrent data streams
    * MRI Data
    * Clinical Data
  
## Requirements

The following requirements are contained within the requirements.yml file. To install them just run:
```
$ conda env create -f environment.yml
```
  
| Library      | Version     | Purpose     |
| :------------- | :----------: | -----------: |
|  Pytorch | 1.11.0   | Deep learning library for tensor transformations    |
| Torchvision | 0.12.0 | library required by pytorch |
| Numpy  | 1.18.1 | Datascience library for creating and manipulating arrays in python  |
| sklearn | 0.23 | Machine learning library used to output area under the curve and other stats. |
| matplotlib | 3.1.2 | A visualization library required by sklearn to generate the 2D graphs. |
| Pandas | 0.25.3 | Datascience library for working with tabular data |
| Nibabel  | 3.0.0 | Library for loading in MRI scans into python  |
| Tqdm  | 4.42.1 | library for outputting progress to the console.  |


## Data

You must place the data folder in a directory above the project folder. The data folder must contain a csv file with the clinical data and a folder containing the MRI images for each class. The data can be obtained from the ADNI website. The data requirements are described in more detail in the wiki but to sumarize the clinical vector must be of length 21 and the mri scans have to be of dimension 110x110x110. Alternatively you can download a zip from this [link](https://drive.google.com/file/d/1QhupPIg9UWU7MkoU9JQnqj5tbf4_fdTF/view?usp=sharing) of my preprocessed dataset. 

## How to use
1. Place your data in the directory above. (See [wiki](https://github.com/McSpooder/camull_net/wiki/Data-Requirements)) 
2. Download the pretrained [weights and database](https://drive.google.com/drive/folders/1lXsZymYNzyt185fBtLR2ZdEsLct4igHQ?usp=sharing) and put it in a directory above. 
2. Run "python camull_cli.py"
3. First train a model on the task NC vs AD.
4. Evaluate the model if you wish.
5. Then load the model and use it to learn the task sMCI v pMCI.
7. Look in the save folder for logs and graphs (..\graphs\, ..\logs\).

### Inference
Once the model has been trained and you are happy with its perfomance after reviewing the graphs and logs you can use it for predictions. The camull_cli script allows you to use load the trained model for predictions.

To read an explanation on this checkout this medium article. 
https://medium.com/@hextra_19712/deep-learning-for-alzheimers-classification-57611161e442?sk=ebc8035cead352f51f7d0e105a69a6c7

![](https://i.gyazo.com/1b655be7b33cff82372088abdbf8eb4c.png)

## Extras
[alt](https://miro.medium.com/max/1000/1*guvHPIlisovNCCltm5W8-w.png "blocks")
[alt](https://miro.medium.com/max/700/1*hbOeKu1qpQXQYA1RNWzkJA.png "architecture")

