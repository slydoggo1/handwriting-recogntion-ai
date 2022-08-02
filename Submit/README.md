# project1-team-27-new

Handwriting Recognizer

Our handwriting recognizer is an application that allows the user to pick from two neural networks to train using the EMNIST dataset. The accuracy of the model is displayed to the user. The user can then save this trained model and load it back for later use. This trained model can be used to make predictions on images downloaded by the user.

Getting started

1. Install Python 3.9.12
2. Open the terminal and change the current directory to the folder containing the requirements.txt file
3. Run pip install ~r requirements.txt in the terminal. This installs all required python packages
4. Change the current directory to the scripts folder
5. Run main.py to open the main GUI
6. Select the batch size amount - This is how much images in the dataset are processed before the model is updated. (We recommend between 32 and 128 for the batch        size) 
7. Click the 'Download Dataset' button to begin downloading the EMNIST dataset to your local drive.
8. Select which model to train from the Select NN box
9. Customise these model hyper-parameters to your liking:
    Number of epochs - This is the number of times the model loops over the training data. (We recommend between 3 and 5 if not using cuda. If you don't mind waiting,                        epochs is a good amount for training)
    Train/validation ratio - This is the ratio in which the total training dataset will be split into for training and validating. (We recommend between 0.1 and 0.3. 
                             This means the validation data makes up between 10% and 30% of the total training data.
10. Click the 'Train Model' button to begin training the selected model
11. Click the 'Save Model' button to save the current model after inputting a model name in the Model Name box if you are satisfied with the accuracy. Otherwise, a new model with the same or different parameters can be trained again by repeating steps 6 to 10

Models

Our application features two different neural networks to select from.

Standard DNN:

Our first neural network is a standard neural network that consists of 5 linear layers. The input of this network takes in a flattened 28 x 28 pixel greyscale image. Here is a summary of the input, hidden and output layers of our model:
Linear Layer 1: 784 input, 660 output
Linear Layer 2: 660 input, 480 output
Linear Layer 3: 480 input, 350 output
Linear Layer 4: 350 input, 120 output
Linear Layer 5: 120 input, 62 output
All layers are activated through ReLU activation. The output is a tensor of the relative probabilities of each of the 62 classes on the predicted image. This model achieves ≈ 84% accuracy after 10 epochs at a learning rate of 0.01 and momentum of 0.5. We used 10% of the training data for validation.

LeNet5 CNN:

Our next neural network is a convolutional neural network based on the LeNet-5 which was one of the first CNN's used for recognizing machine and handwritten characters. The reason this was chosen was because it is one the most simple CNN's for image classification. This model takes in a flattened 32 x 32 pixel greyscale image. Here is a summary of the input, hidden and output layers of our model:
Convolutional Layer 1: 2D, 1 input, 6 output, 5x5 kernel size, 1 stride, 0 padding
Convolutional Layer 2: 2D, 6 input, 16 output, 5x5 kernel size, 1 stride, 0 padding
Convolutional Layer 3: 2D, 16 input, 120 output, 5x5 kernel size, 1 stride, 0 padding
Linear Layer 1: 120 input, 84 output
Linear Layer 2: 82 input, 62 output
All layers are activated through Tanh activation and there are two average pooling layers between both convolutional layers. This model achieves ≈ 85% accuracy after 6 epochs at learning rate of 0.01 and momentum of 0.5. We used 20% of the training data for validation.

