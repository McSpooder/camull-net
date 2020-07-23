# camull-net
An implementation of the Cambridge parameter efficient 3D convolutional network for the classification of Alzheimer's disease in pytorch. The architecture is described in detail in this [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191930031X "paper").

The code was developed as part of my dissertation at the university of Hull. You can read more about it in this medium [link](https://medium.com/@hextra_19712/deep-learning-for-alzheimers-classification-57611161e442 "article").

## Features

* Neural Net
  * 3D seperable convolution to reduce number of parameters and hence overfitting.
  * Multiple concurrent data streams
    * MRI Data
    * Clinical Data
* Program
  * Saves and loads model
  * Supports k-fold cross validation
  * Logs the train and test loss
  * Outputs Area under the ROC curve for each k-fold
  * Saves each model under a unique identifier
  
## Requirements
  
| Library      | Version     | Purpose     |
| :------------- | :----------: | -----------: |
|  Pytorch | 1.4.0   | Deep learning library for tensor transformations    |
| Numpy  | 1.18.1 | Datascience library for creating and manipulating arrays in python  |
| Nibabel  | 3.0.0 | Library for loading in MRI scans into python  |
| Tqdm  | 4.42.1 | library for outputting progress to the console.  |



To read an explanation on this checkout this medium article. 
https://medium.com/@hextra_19712/deep-learning-for-alzheimers-classification-57611161e442?sk=ebc8035cead352f51f7d0e105a69a6c7

!(image)[https://i.gyazo.com/1b655be7b33cff82372088abdbf8eb4c.png]
