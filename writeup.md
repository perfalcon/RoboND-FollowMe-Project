### Project: Follow Me

### Write Up
In this project created a Deep Neural Network - a Fully Convolution Dee[Neural Network to train the images from the simulator to identify a specific target herein after is referred to as "hero". Classified the hero from others using two different colors of pixels one for hero and other for all others using the sematic segmentation.

Steps followed :
1) Set up the environment 
2) Get Data
3) Train the Neural Network to acheive more than .40 accuracy.
5) Test the trained data in the Simulator.

## Setup Environment

Setup AWS Instance:
```
  https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f
```

Download Simulator
```
https://github.com/udacity/RoboND-DeepLearning-Project/releases/tag/v1.2.2
```

Get the Project Frame work from github:

```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

Download the data

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

Achieve the Accuracy of .40:

Created FCNN by adjusting the hyper paramters ( epochs, batches, steps for epoch, validations for epoch)

  Get the data(images) for training and validation from the simulator with and without the hero.
  Train the Model on AWS udacity's ami.
  Verify the final score for accuracy
  Running the model in the simulator to follow the hero.

Follow Me in Simulator
![follow me](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/follow-me-sim.PNG)

## Data:
Used the data provided by the udacity and gather some more data from simulator with following aspects:
  1) Following the hero in dense crowd
  2) Patroling only on hero
  3) Quad on standard patrol

## Model ##
This FCN model consists of  encoders layers, 1 x 1 convolution  and decoder  layers  to build the semantic segmentation. 
### Step 1: ###
*Encode Layer:*
It does the separable convolution operation with ReLU activation and then batch normalization.

*Separable Convolution:*

...The Separable convolution is a technique that reduces the number of parameters needed.
The reduction in the parameters make separable convolutions quite efficient with improved runtime performance and are also, as a result, useful for mobile applications. They also have the added benefit of reducing overfitting to an extent, because of the fewer parameters.

*Batch Normalizaion:*

...Batch normalization is the process of normalizing the inputs to layers within the network, instead of just normalizing the inputs to the network. It's called "batch" normalization because during training, we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch. This is has following advantages :
 
  * Networks train faster
  * Allows higher Learning rates
  * Simplifies the creation of  deeper networks
  * Provides a bit of regularization.
  
This encoder block calls the `separable_conv2d_batchnorm` :
  `def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer`
 
The `separable_conv2d_batchnorm` calls the `SeparableConv2DKeras` and then does the batch normalization : `layers.BatchNormalization`
 `def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer`


asdfa
  


