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
*Encode the Layer/s:*
Encoding does the separable convolution operation with ReLU activation and then batch normalization.

*Separable Convolution:*

The Separable convolution is a technique that reduces the number of parameters needed.
The reduction in the parameters make separable convolutions quite efficient with improved runtime performance and are also, as a result, useful for mobile applications. They also have the added benefit of reducing overfitting to an extent, because of the fewer parameters.

*Batch Normalizaion:*

Batch normalization is the process of normalizing the inputs to layers within the network, instead of just normalizing the inputs to the network. It's called "batch" normalization because during training, we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch. This is has following advantages :
 
  * Networks train faster
  * Allows higher Learning rates
  * Simplifies the creation of  deeper networks
  * Provides a bit of regularization.
  
This encoder block calls the `separable_conv2d_batchnorm` :
 ```
  def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer`
 ```
The `separable_conv2d_batchnorm` calls the `SeparableConv2DKeras` and then does the batch normalization : `layers.BatchNormalization`
 ```
 def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer`
```

### Step 2: ###
 Then apply the 1 x 1 convolution with `conv2d_batchnorm` which does a 1 x 1 convolution with ReLU activation and then the batch normalization.
 ```
 def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
 ```
  
### Step 3: ###
*Decode the Layer/s:*
In this step following operations are performed:
* Bilinear UpSampling:

This is one of the way to implement the upsampling.
This technique utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.
This is implemented by `bilinear_upsample`
```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
 
* A layer concatenation step

This step is similar to skip connections. we will concatenate the upsampled small_ip_layer and the large_ip_layer.
This is implemented by `layers.concatenate`

* Additional Separable Convolutions

This is done to extract some more spatial information from prior layers and implemented by `separable_conv2d_batchnorm`


### FCN Model:###
Created the FCN with three encoders, then a 1x1 convolution, then three decoders and then apply a convolution with an activation of softmax.

```
def fcn_model(inputs, num_classes):
    # Encoder Layers
    layer1 = encoder_block(inputs, 32, strides=2)
    layer2 = encoder_block(layer1, 64, strides=2)
    layer3 = encoder_block(layer2, 64, strides=2)
    # 1x1 Convolution layer using conv2d_batchnorm().
    condv2d_batchnormed = conv2d_batchnorm(layer3, 64, kernel_size=1, strides=1)
    # Addedsame number of Decoder Blocks as the number of Encoder Blocks
    layer4 = decoder_block(condv2d_batchnormed,layer2,64)
    layer5 = decoder_block(layer4,layer1,64)
    x = decoder_block(layer5,inputs,32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## Training ##
Trained the model in the AWS Udacity's AMI instance with provided data from udacity and collected data from simulator.

*HyperParameters*
* batch_size: number of training samples/images that get propagated through the network in a single pass.
* num_epochs: number of times the entire training dataset gets propagated through the network.
* steps_per_epoch: number of batches of training images that go through the network in each epoch.
* validation_steps: number of batches of validation images that go through the network in each epoch. 
* workers: maximum number of processes to spin up.

