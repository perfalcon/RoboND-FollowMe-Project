### Project: Follow Me

### Write Up
In this project created a Deep Neural Network - a Fully Convolution Deep Neural Network to train the images from the simulator to identify a specific target (called hero). Classified the hero from others using two different colors of pixels one for hero and second color for all others using the sematic segmentation.

Steps followed :
1) Set up the environment 
2) Get Data
3) Train the Neural Network to acheive more than .40 final score.
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


## Data Collection
Used the data provided by the udacity and gather some more data from simulator with following aspects:
  1) Following the hero in dense crowd
  2) Patroling only on hero
  3) Quad on standard patrol

## Model ##
This FCN model consists of  encoders layers, 1 x 1 convolution  and decoder  layers  to build the semantic segmentation. 
Below is the architecture for the model:

![FCN Model](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/fcn-model.PNG)

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


### FCN Model ###
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

Arrived the required accuracy for the model with the following values for the HyperParameters based on the pervious exercises and trails in this model.
```
learning_rate = 0.005
batch_size = 64
num_epochs = 20
steps_per_epoch = 400
validation_steps = 50
workers = 2
```
Below is the Training curves images:

| Training curve at Epoch -2        | Training curve at Epoch -20  |
| ------------- |:-------------:|
| ![epoch2](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/graph-epoch-2.PNG) | ![epoch20](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/graph-epoch-20.PNG) |


## Prediction ##
The predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well the model is doing under different conditions.

There are three different predictions available from the helper code provided:
- **patrol_with_targ**: Test how well the network can detect the hero from a distance.
- **patrol_non_targ**: Test how often the network makes a mistake and identifies the wrong person as the target.
- **following_images**: Test how well the network can identify the target while following them.

| Prediction following the Target        | Prediction without Target  |
| ------------- |:-------------:|
| ![follow me](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/following-target.PNG) | ![No Target](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/patrol-without-target.PNG) |

| Prediction while the Target at a distance       |
| ------------- | 
| ![Target a Distance](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/patrol-with-target.PNG) | 

## Evaluation ##
The scores for the above predictions :

**Scores for while the quad is following behind the target**
```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9953953783924273
average intersection over union for other people is 0.3687066701416392
average intersection over union for the hero is 0.9101514501130293
number true positives: 539, number false positives: 0, number false negatives: 0
```
**Scores for images while the quad is on patrol and the target is not visable**
```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.988607098705327
average intersection over union for other people is 0.7642755767950153
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 63, number false negatives: 0

```
**Scores for images while quad is on patrol and the target is at a distance**
```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9966889879590972
average intersection over union for other people is 0.4561759249143874
average intersection over union for the hero is 0.2484021697518587
number true positives: 143, number false positives: 1, number false negatives: 158

```
**Sum all the true positives, etc from the three datasets to get a weight for the score**
```
Weight = 0.754424778761062
```
**The IoU for the dataset that never includes the hero is excluded from grading**
```
IOU 0.579276809932
```
**Final grade score is `0.437020779175`**


**Run the model in the simulator**
Follow Me in Simulator
![follow me](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/images/follow-me-sim.PNG)


[Code for model](https://github.com/perfalcon/RoboND-FollowMe-Project/blob/master/scripts/model_training.ipynb)


[Weights Trained weights](https://github.com/perfalcon/RoboND-FollowMe-Project/tree/master/weights)


## Imporvements ##
* Want to build and train with different set of images for other types of object like, animals, moving vehicles 

