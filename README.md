### SVA_CNN
This is Keras based code for Semblance Velocity Analysis using Convolutional Neural Networks.


### Environment
-  Python 3.6
-  Tensorflow 1.13
-  Keras 2.2.4
-  Cuda 10.1
-  Cudnn 7.5 

### CNN models we adopt
-  Alexnet [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)
-  VGG16 [paper](https://arxiv.org/abs/1409.1556)
-  LeNet5 [Link](http://yann.lecun.com/exdb/lenet/)

### Description 
1. **1_base_model_training.ipynb**
It is for the base model training with custom synthetic dataset. After the training, we test the trained model with another data which were not used for the training. 



2. **2_transfer_learning.ipynb**
It is for transfer learning. We update the base model with a small portion of Marmousi dataset so that the updated model can predict entire semblance of Marmousi. 


