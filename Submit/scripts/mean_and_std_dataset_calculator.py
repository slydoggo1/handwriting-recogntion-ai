
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils import data

#train_data = datasets.EMNIST(root="./dataset", split="byclass", train=True, download=True, transform=transforms.ToTensor()) # change this to split into train/validation data
#test_data = datasets.EMNIST(root="./dataset", split="byclass", train=False, download=True, transform=transforms.ToTensor())

#print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(test_data.data.float().mean() / 255, test_data.data.float().std() / 255))

input = input("Enter model name here: ")
path = "./saved_models/" + input + ".pth"
print(path)

