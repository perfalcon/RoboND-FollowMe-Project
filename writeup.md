### Project: Follow Me

### Write Up
In this project created a Deep Neural Network - a Fully Convolution Dee[Neural Network to train the images from the simulator to identify a specific target herein after is referred to as "hero". Classified the hero from others using two different colors of pixels one for hero and other for all others using the sematic segmentation.

Steps followed :
1) Set up the environment 
2) Get Data
3) Train the Neural Network to acheive more than .40 accuracy.
5) Test the trained data in the Simulator.

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

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
