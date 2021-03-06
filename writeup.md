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

![FCN Model](./images/fcn-model.PNG)

### Step 1: ###
*Encoder*

The Encoder extracts the features by applying the 2d convolutions and the batch normalization.

*Encode the Layer/s:*
Encoding does the separable convolution operation with ReLU activation and then batch normalization.

*Separable Convolution:*

The Separable convolution is a technique that reduces the number of parameters needed.
The reduction in the parameters make separable convolutions quite efficient with improved runtime performance and are also, as a result, useful for mobile applications. They also have the added benefit of reducing overfitting to an extent, because of the fewer parameters.

*Batch Normalizaion:*

Batch normalization is the process of normalizing the inputs to layers within the network, instead of just normalizing the inputs to the network. It's called "batch" normalization because during training, we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch. This has following advantages :
 
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
 
The 1 by 1 convolution preserves the spatial information, but when the fully connected layer is applied there is a loss of spatial information, because no information about the location of the pixels is preserved. With this we can know where is the specific object in the Image.

We know 'there is a cat in the scene(Image)' and with this 1x1 Convolution we will know 'where the cat is in the scene(Image)'

 
 ```
 def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
 ```
  
### Step 3: ###
*Decoder*

Decoder upscales the output such that it is same size of the Image by retaining the spatial information.

This process consists of the following techniques:
* UpSampling through Bilinear UpSampling
* Skip Connections

Implemented through `decoder_block`

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    sampled_layer = bilinear_upsample(small_ip_layer)
    #Concatenate the upsampled and large input layers using layers.concatenate
    cat_layer = layers.concatenate([sampled_layer, large_ip_layer])
    # Add some number of separable convolution layers
    conv_layer = separable_conv2d_batchnorm(cat_layer, filters)
    output_layer = separable_conv2d_batchnorm(conv_layer, filters)
    return output_layer
 ```

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
 
* Skip Connections

Skip connections is a great way to retain some of the finer details from the previous layers as we decode or upsample the layers to the original size. 

It is a process of combining the output with non-adjacent layers.

As I know off, this can be implemented in two ways - element wise addition of two layers and concatenation of two layers.
The element wise addition of two layers needs to have same depth of the two layers.
The Concatenation of two layers, provides a flexibility, that it need not be the same depth of the layers.

Here concatenated the upsampled small_ip_layer and the large_ip_layer.
`layers.concatenate([sampled_layer, large_ip_layer])`

* Additional Separable Convolutions

This is done to extract some more spatial information from prior layers and implemented by `separable_conv2d_batchnorm`



### FCN Model ###
*Fully Convolution Layer*
This consists of three techniques:
1) Replace Fully Connected Layers with 1 x 1 Convolution Layer ( to get the spatial information - location of the pixels).
2) UpSampling the output to the same size of the Image.
3) Skip connections to retain some of the finer details from the pervious layers.


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

Arrived the required final score for the model with the following values for the HyperParameters based on the pervious exercises and trails in this model.

Started trianing the model with a learning rate of 0.05, batch size of 20, number of epoch of 10, steps per epoch 100 and validation steps of 20 got a score of 0.30 and then increased batch size to 40, saw increase in score to 0.35, then decreased the learning rate to 0.005 and increased number of epochs to 20, there is an increase of score to 0.41, then increased the batch size to 64 and steps per epoch at 400, there is slight increase in the score to 0.42 and then increased the validation steps to 50, saw slight incresae in the score to 0.43

As per the project requirement I took the this final score of 0.43.

On overall, i see there is a significant change in the score when the number of epochs increased to 20 and reduced the learning rate to 0.005

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
| ![epoch2](./images/graph-epoch-2.PNG) | ![epoch20](./images/graph-epoch-20.PNG) |


**Trails**

|HyerPer Parameters   | Final Score   |Training Curve at Epoch -20   |
|---|---|---|
| ![Run1](./images/parameters-pics/Run1.PNG)|0.30291113035|![Run1](./images/parameters-pics/Run1-Graph.png)   |
| ![Run2](./images/parameters-pics/Run2.PNG)|0.352162249037|![Run2](./images/parameters-pics/Run2-Graph.png)   |
| ![Run3](./images/parameters-pics/Run3.PNG)|0.419950640633|![Run3](./images/parameters-pics/Run3-Graph.png)   |
| ![Run4](./images/parameters-pics/Run4.PNG)|0.423943902236|![Run4](./images/parameters-pics/Run4-Graph.png)   |
| ![Run5](./images/parameters-pics/Run5.PNG)|0.437020779175|![Run5](./images/parameters-pics/Run5-Graph.png)   |


## Prediction ##
The predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well the model is doing under different conditions.

There are three different predictions available from the helper code provided:
- **patrol_with_targ**: Test how well the network can detect the hero from a distance.
- **patrol_non_targ**: Test how often the network makes a mistake and identifies the wrong person as the target.
- **following_images**: Test how well the network can identify the target while following them.

| Prediction following the Target        | Prediction without Target  |
| ------------- |:-------------:|
| ![follow me](./images/following-target.PNG) | ![No Target](./images/patrol-without-target.PNG) |

| Prediction while the Target at a distance       |
| ------------- | 
| ![Target a Distance](./images/patrol-with-target.PNG) | 

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
![follow me](./images/follow-me-sim.PNG)


[Code for model](./scripts/model_training.ipynb)


[Weights Trained weights](./weights)


**Using this Model for Other Objects**

We cannot use this model for identifying other objects like dog, cat, car, etc., as we have trained this model specically to recognize the human objects ( hero , other people and everything is one segment). Inorder to identify other objects, we have to gather those data(images), classify them accordingly, train and segment them.



## Imporvements ##
* Want to build and train with different set of images for other types of object like, animals, moving vehicles 

