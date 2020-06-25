# DNN-classifire-using-TensorFlow_2.0
This is neural network with one input layer, two hidden layers and ten output layers. It uses TensorFlow2.0 and Keras as arhetecture of neural network.
## How it works ?

At first il loads MINST Fashion dataset from Tensorflow and split it into train and test labels.Model uses [Adam optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) and saves trained model evry epoch in checkpoint directory that creates at the begining. Network has 6 layers:
1. First layer is input layer that reshapes input in 28x28 pixels
2. Second layer is first hidden layer with 200 neurons, it uses [ELU activation function](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/) and [Variance scaling initalizer](https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)
3. Third layer is [Dropout function](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)
4. Fourth layer is second hidden layer, it's preatty same as first hidden layer but has only 100 neurons
5. Fift layer is [Batch normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
6. Last layer is output layer that use [Softmax activation function](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d) 
Model uses [Adam optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) and saves trained model evry epoch in checkpoint directory that creates at the begining.

## How can I use it?

To use model you need installed [Python](https://www.python.org/downloads/) and [TensorFlow](https://www.tensorflow.org/install) minimal version 2.0
