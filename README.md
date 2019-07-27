# SVA_CNN
Semblance Velocity Analysis using Convolutional Neural Networks.


# Environment
-  Python 3.6
-  Tensorflow 1.13
-  Cuda 10.1
-  Cudnn 7.5 

# Description 
1. **1_base_model_training.ipynb**
It is for the base model training with custom synthetic dataset. After the training, we test the trained model with another data which were not used for the training. 



2. **2_transfer_learning.ipynb**
It is for transfer learning. We update the base model with a small portion of Marmousi dataset so that the updated model can predict entire semblance of Marmousi. 


