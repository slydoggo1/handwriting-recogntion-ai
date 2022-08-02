# This file is for testing the AI side of the program.

import torch
import torchvision
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision import datasets, transforms
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image  
import PIL.ImageOps  

'''Testing a standard NN and CNN to train the EMNIST dataset'''

# Test 1: Standard neural network with 3 linear layers, reLU activation - 83% accuracy after 9 epochs, 128 batch size
# Test 2: Standard neural network with 5 linear layers, reLU activation - 84% accuracy after 11 epochs, 128 batch size
# Test 3: Standard neural network with 5 linear layers, reLU activation - 84% accuracy after 5 epochs, 64 batch size (same hidden layers as test 2)
# Test 4: Standard neural network with 5 linear layers, reLU activation - 85% accuracy after 5 epochs, 32 batch size (same hidden layers as test 2)
# Test 5: Convolutional neural network with 2 convolutional layers, 2 linear layers, reLU activation - 82% accuracy after 3 epochs, 64 batch size, 10% validation data
# Test 6: LeNet5 nerural network - 84% accuracy after 3 epochs, 64 batch size, 20% validation data
# Test 7: LeNet5 nerural network - 85% accuracy after 6 epochs, 32 batch size, 20% validation data

device = 'cuda' if cuda.is_available() else 'cpu'
#print(f'Training EMNIST Model on {device}\n{"=" * 44}')

# DNN hyper-parameters (These will be set by user)
num_of_epochs = 1
batch_size = 64
validation_to_train_ratio = 0.1

# Transform function to normalize data
transform_EMNIST = transforms.Compose(
    [ToTensor(), Normalize((0.1736,), (0.3317,))])
resize_EMNIST = transforms.Compose(
    [Resize((32, 32)), ToTensor(), Normalize((0.1736,), (0.3317,))])


# Download EMNIST Dataset 26 letters (lower and uppercase) and 10 digits (0 to 9) for a total of 62 unbalanced classes. Another download of the dataset is done but for
# resizing the images to 32 x 32 pixels instead of 28 x 28. This is because the LeNet 5 model takes in an input of 32 x 32 images.
all_train_data = datasets.EMNIST(
    root="./dataset", split="byclass", train=True, download=True, transform=transform_EMNIST)
all_train_data_resized = datasets.EMNIST(
    root="./dataset_resized", split="byclass", train=True, download=True, transform=resize_EMNIST)

validation_data_size = round(validation_to_train_ratio * len(all_train_data))
train_data_size = len(all_train_data) - validation_data_size
train_data, validation_data = data.random_split(all_train_data, [
                                                train_data_size, validation_data_size])  # Split all the training data into train
# and validation according to ratio specified by user

train_data_resized, validation_data_resized = data.random_split(
    all_train_data_resized, [train_data_size, validation_data_size])

test_data = datasets.EMNIST(root="./dataset", split="byclass",
                            train=False, download=True, transform=transform_EMNIST)
test_data_resized = datasets.EMNIST(
    root="./dataset_resized", split="byclass", train=False, download=True, transform=resize_EMNIST)

# Move data through the data loader
train_dl = data.DataLoader(train_data, batch_size, shuffle=True)
validation_dl = data.DataLoader(validation_data, batch_size, shuffle=True)
test_dl = data.DataLoader(test_data, batch_size, shuffle=True)

# Data loaders of resized images used for LeNet5 training
train_dl_r = data.DataLoader(train_data_resized, batch_size, shuffle=True)
validation_dl_r = data.DataLoader(validation_data_resized, batch_size, shuffle=True)
test_dl_r = data.DataLoader(test_data_resized, batch_size, shuffle=True)

# Architecture of a standard neural network
class StandardNeuralNet(nn.Module):

    def __init__(self):
        super(StandardNeuralNet, self).__init__()
        # 5 linear layers, 784 input neurons (28 x 28 pixel image), 62 output neurons (62 classes)
        self.linear1 = nn.Linear(784, 660)
        self.linear2 = nn.Linear(660, 480)
        self.linear3 = nn.Linear(480, 350)
        self.linear4 = nn.Linear(350, 120)
        self.linear5 = nn.Linear(120, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

# Architecture of a custom convolutional neural network taken from https://github.com/sksq96/pytorch-summary


class ConvolutionalNeuralNet(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.convolution1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convolution2 = nn.Conv2d(10, 20, kernel_size=5)
        self.convolution_drop = nn.Dropout2d()
        self.linear1 = nn.Linear(320, 120)
        self.linear2 = nn.Linear(120, 62)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 2))
        x = F.relu(F.max_pool2d(self.convolution_drop(self.convolution2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        return F.log_softmax(x)

# Architecture of a LeNet 5 neural network modified to classify 62 classes
class LeNet5NeuralNet(nn.Module):

    def __init__(self):
        super(LeNet5NeuralNet, self).__init__()                             
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 62)
        self.avgpool = nn. AvgPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.avgpool(x)
        x = self.tanh(self.conv2(x))
        x = self.avgpool(x)
        x = self.tanh(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)

        return x

# This function will save the current model if trained and return the path to the saved model
def saveModel(model):
    # This will be changed to a user input later
    model_name = "cnn_model_1"
    # Use user input to create path to saved model
    path = model_name + ".pth"
    torch.save(model.state_dict(), path)

    return path

# This function will load and return one of the saved models for predicting
def loadModel(model_type, path):
    # Choose which model to recreate
    if model_type == "Standard":                        
        model = StandardNeuralNet()
    else:
        model = LeNet5NeuralNet()

    # Load back in parameters of saved model
    model.load_state_dict(torch.load(path))

    return model

# This function will take in an input and return the predicted and expected class of the input. Input is one image of the testing dataset
def predict_from_dataset(model, input, target, class_mapping):
    # Change model into testing mode
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        #predicted_index = predictions[0].argmax(0)
        _, predicted_index = torch.max(predictions.data, 1)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

# This function will take make a prediction on a downloaded image created by the user
def predict_custom_image(model, image_transform, path, class_mapping):
    # Change model into testing mode
    model = model.eval()
    # Open image and transform it, ready to be sent to the model
    image = Image.open(path)                    
    image= image.convert('1') 
    image = image_transform(image).float()
    image = image.unsqueeze(0)
    # Input the image into the model and make prediction
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print("Predicted character: " + class_mapping[predicted])

# Mapping of 62 classes
class_mapping = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]

# This function takes in the current epoch number and begins inputting data into the model
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dl_r):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dl_r.dataset),
                100. * batch_idx / len(train_dl), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_dl_r:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_dl.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(validation_dl_r.dataset)} '
          f'({100. * correct / len(validation_dl_r.dataset):.0f}%)')


if __name__ == '__main__':

    # Create a LeNet5 model
    model = LeNet5NeuralNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Begin training model
    since = time.time()
    for epoch in range(1, num_of_epochs + 1):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    # Save and load trained model
    path = saveModel(model)

    #path = "cnn_model_1.pth" 

    loaded_model = loadModel("CNN", path)

    # Make a prediction on downloaded image made by user  
    predict_custom_image(loaded_model, resize_EMNIST, "test_image_g.jpg", class_mapping)

    # Get an image from the testing dataset for prediction
    input, target = next(iter(test_dl_r))
    #input, target = test_data_resized[0][0], test_data_resized[0][1]

    predicted, expected = predict_from_dataset(loaded_model, input, target, class_mapping)

    print(f"Predicted: '{predicted}', expected: '{expected}'")



