from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
import time
import sys
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

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 588)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.HomePg = QtWidgets.QWidget(self.centralwidget)
        self.HomePg.setGeometry(QtCore.QRect(-10, -10, 821, 621))
        self.HomePg.setStyleSheet("QWidget#HomePg{\n"
"background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(151, 219, 255, 255), stop:1 rgba(255, 255, 255, 255))}")
        self.HomePg.setObjectName("HomePg")

        self.Train_Button = QtWidgets.QPushButton(self.HomePg)
        self.Train_Button.setGeometry(QtCore.QRect(40, 340, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Train_Button.setFont(font)
        self.Train_Button.clicked.connect(self.start_training)
        self.Train_Button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.Train_Button.setObjectName("Train_Button")
        #self.Train_Button.clicked.connect(self.start_training)

        #exit button that terminates app
        self.exit_button = QtWidgets.QPushButton(self.HomePg)
        self.exit_button.setGeometry(QtCore.QRect(700, 490, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.exit_button.clicked.connect(self.exit_app)
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.exit_button.setObjectName("exit_button")

        self.save_button = QtWidgets.QPushButton(self.HomePg)
        self.save_button.setGeometry(QtCore.QRect(500, 490, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.save_button.clicked.connect(self.saveModel)
        self.save_button.setFont(font)
        self.save_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.save_button.setObjectName("save_button")

        self.load_button = QtWidgets.QPushButton(self.HomePg)
        self.load_button.setGeometry(QtCore.QRect(600, 490, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.load_button.clicked.connect(self.loadModel)
        self.load_button.setFont(font)
        self.load_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.load_button.setObjectName("save_button")

        self.title = QtWidgets.QLabel(self.HomePg)
        self.title.setGeometry(QtCore.QRect(220, 10, 450, 70))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.title.setFont(font)
        self.title.setObjectName("title")

        self.epoch_label = QtWidgets.QLabel(self.HomePg)
        self.epoch_label.setGeometry(QtCore.QRect(40, 90, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.epoch_label.setFont(font)
        self.epoch_label.setObjectName("epoch_label")

        self.batch_label = QtWidgets.QLabel(self.HomePg)
        self.batch_label.setGeometry(QtCore.QRect(200, 90, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.batch_label.setFont(font)
        self.batch_label.setObjectName("batch_label")

        self.DownloadBar = QtWidgets.QProgressBar(self.HomePg)
        self.DownloadBar.setGeometry(QtCore.QRect(40, 200, 311, 23))
        self.DownloadBar.setStyleSheet("")
        self.DownloadBar.setProperty("value", 0)
        self.DownloadBar.setObjectName("DownloadBar")

        self.train_cancel_button = QtWidgets.QPushButton(self.HomePg)
        self.train_cancel_button.setGeometry(QtCore.QRect(230, 340, 110, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.train_cancel_button.setFont(font)
        self.train_cancel_button.clicked.connect(self.stop_training)
        self.train_cancel_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.train_cancel_button.setObjectName("train_cancel_button")

        self.epoch_spin = QtWidgets.QSpinBox(self.HomePg, value=12)
        self.epoch_spin.setGeometry(QtCore.QRect(50, 120, 61, 22))
        self.epoch_spin.setObjectName("epoch_spin")

        self.batch_spin = QtWidgets.QSpinBox(self.HomePg, value=64)
        self.batch_spin.setGeometry(QtCore.QRect(210, 120, 61, 22))
        self.batch_spin.setObjectName("batch_spin")

        self.TrainBar = QtWidgets.QProgressBar(self.HomePg)
        self.TrainBar.setGeometry(QtCore.QRect(40, 300, 311, 23))
        self.TrainBar.setProperty("value", 0)
        self.TrainBar.setObjectName("TrainBar")

        self.download_label = QtWidgets.QLabel(self.HomePg)
        self.download_label.setGeometry(QtCore.QRect(40, 170, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.download_label.setFont(font)
        self.download_label.setObjectName("download_label")

        self.train_label = QtWidgets.QLabel(self.HomePg)
        self.train_label.setGeometry(QtCore.QRect(40, 270, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_label.setFont(font)
        self.train_label.setObjectName("train_label")

        self.ratio_label = QtWidgets.QLabel(self.HomePg)
        self.ratio_label.setGeometry(QtCore.QRect(700, 160, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.ratio_label.setFont(font)
        self.ratio_label.setObjectName("ratio_label")

        self.train_slider = QtWidgets.QSlider(self.HomePg)
        self.train_slider.setMaximum(9)
        self.train_slider.setMinimum(1)
        self.train_slider.setGeometry(QtCore.QRect(520, 130, 191, 22))
        self.train_slider.setOrientation(QtCore.Qt.Horizontal)
        self.train_slider.setObjectName("train_slider")
        self.train_slider.valueChanged.connect(self.ratio)

        self.train_ratio_label = QtWidgets.QLabel(self.HomePg)
        self.train_ratio_label.setGeometry(QtCore.QRect(500, 100, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_ratio_label.setFont(font)
        self.train_ratio_label.setObjectName("train_ratio_label")

        self.validation_label = QtWidgets.QLabel(self.HomePg)
        self.validation_label.setGeometry(QtCore.QRect(670, 100, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.validation_label.setFont(font)
        self.validation_label.setObjectName("validation_label")

        self.accuracy_label = QtWidgets.QLabel(self.HomePg)
        self.accuracy_label.setGeometry(QtCore.QRect(500, 320, 91, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.accuracy_label.setFont(font)
        self.accuracy_label.setObjectName("accuracy_label")

        self.select_combo = QtWidgets.QComboBox(self.HomePg)
        self.select_combo.setGeometry(QtCore.QRect(500, 230, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_combo.setFont(font)
        self.select_combo.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.select_combo.setObjectName("select_combo")
        self.select_combo.addItem("EMNIST DNN")
        self.select_combo.addItem("Lenet5 CNN")

        self.Select_nn_label = QtWidgets.QLabel(self.HomePg)
        self.Select_nn_label.setGeometry(QtCore.QRect(500, 200, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Select_nn_label.setFont(font)
        self.Select_nn_label.setObjectName("Select_nn_label")

        self.download_time_label = QtWidgets.QLabel(self.HomePg)
        self.download_time_label.setGeometry(QtCore.QRect(150, 170, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.download_time_label.setFont(font)
        self.download_time_label.setObjectName("download_time_label")

        self.download_button = QtWidgets.QPushButton(self.HomePg, clicked =lambda: self.download_model())
        self.download_button.setGeometry(QtCore.QRect(40, 230, 175, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.download_button.setFont(font)
        self.download_button.clicked.connect(self.start_download)
        self.download_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.download_button.setObjectName("download_button")

        self.download_cancel_button = QtWidgets.QPushButton(self.HomePg)
        self.download_cancel_button.setGeometry(QtCore.QRect(230, 230, 110, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.download_cancel_button.setFont(font)
        self.download_cancel_button.clicked.connect(self.stop_download)
        self.download_cancel_button.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.download_cancel_button.setObjectName("download_cancel_button")

        self.train_ratio = QtWidgets.QLabel(self.HomePg)
        self.train_ratio.setGeometry(QtCore.QRect(500, 160, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_ratio.setFont(font)
        self.train_ratio.setObjectName("train_ratio")

        self.model_name = QtWidgets.QTextEdit(self.HomePg)
        self.model_name.setGeometry(QtCore.QRect(500, 300, 130, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.model_name.setFont(font)
        self.model_name.setObjectName("model_name")

        self.model_name_label = QtWidgets.QLabel(self.HomePg)
        self.model_name_label.setGeometry(QtCore.QRect(500, 275, 90, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.model_name_label.setFont(font)
        self.model_name_label.setObjectName("model_name_label")

        MainWindow.setCentralWidget(self.centralwidget)
        self.actionTrain_Model = QtWidgets.QAction(MainWindow)
        self.actionTrain_Model.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionTrain_Model.setObjectName("actionTrain_Model")

        self.actionQuite = QtWidgets.QAction(MainWindow)
        self.actionQuite.setObjectName("actionQuite")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    device = 'cuda' if cuda.is_available() else 'cpu'

    # This function will save the current model if trained and return the path to the saved model
    def saveModel(model):
        # This will be changed to a user input later
        model_name = "cnn_model_1"
        # Use user input to create path to saved model
        path = model_name + ".pth"
        torch.save(model.state_dict(), path)
        return path

    # This function will load and return one of the saved models for predicting
    def loadModel(self, path):
        # Choose which model to recreate\
        model_type = str(self.select_combo.currentText())
        if model_type == "EMNIST DNN":                        
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

    def exit_app(self):
        quit()

    def ratio(self, value):
        self.ratio_label.setText(str((value/10)))
        global ratio_value
        ratio_value = (value/10)

    def start_training(self):
        self.thread = ThreadClass()
        self.thread.any_signal.connect(self.set_train_value)
        self.thread.start()
        self.Train_Button.setEnabled(False)

    def stop_training(self):
        self.thread.stop()
        self.Train_Button.setEnabled(True)

    def set_train_value(self, val):
        self.TrainBar.setValue(val)

    def start_download(self):
        self.thread = ThreadClass()
        self.thread.any_signal.connect(self.set_download_value)
        self.thread.start()
        self.download_button.setEnabled(False)

    def stop_download(self):
        self.thread.stop()
        self.download_button.setEnabled(True)


    def set_download_value(self, val):
        self.DownloadBar.setValue(val)

    def download_model(self):
        #s_value = str(self.select_combo.currentText())
        b_value = int(self.batch_spin.value())
        r_value = (int(self.train_slider.value())/10)

        transform_EMNIST = transforms.Compose(
            [ToTensor(), Normalize((0.1736,), (0.3317,))])
        resize_EMNIST = transforms.Compose(
            [Resize((32, 32)), ToTensor(), Normalize((0.1736,), (0.3317,))])

        self.download_label.setText("Downloading")
        # Download EMNIST Dataset 26 letters (lower and uppercase) and 10 digits (0 to 9) for a total of 62 unbalanced classes. Another download of the dataset is done but for
        # resizing the images to 32 x 32 pixels instead of 28 x 28. This is because the LeNet 5 model takes in an input of 32 x 32 images.
        all_train_data = datasets.EMNIST(
            root="./dataset", split="byclass", train=True, download=True, transform=transform_EMNIST)
        all_train_data_resized = datasets.EMNIST(
            root="./dataset_resized", split="byclass", train=True, download=True, transform=resize_EMNIST)

        validation_data_size = round(r_value * len(all_train_data))
        train_data_size = len(all_train_data) - validation_data_size
        train_data, validation_data = data.random_split(all_train_data, [ train_data_size, validation_data_size])  
        # Split all the training data into train
        # and validation according to ratio specified by user

        train_data_resized, validation_data_resized = data.random_split(
            all_train_data_resized, [train_data_size, validation_data_size])

        test_data = datasets.EMNIST(root="./dataset", split="byclass", train=False, download=True, transform=transform_EMNIST)
        test_data_resized = datasets.EMNIST(
            root="./dataset_resized", split="byclass", train=False, download=True, transform=resize_EMNIST)

        # Move data through the data loader
        global train_dl_r
        global train_dl
        global validation_dl
        global validation_dl_r
        global test_dl
        global test_dl_r
        train_dl = data.DataLoader(train_data, b_value, shuffle=True)
        validation_dl = data.DataLoader(validation_data, b_value, shuffle=True)
        test_dl = data.DataLoader(test_data, b_value, shuffle=True)

        # Data loaders of resized images used for LeNet5 training
        train_dl_r = data.DataLoader(train_data_resized, b_value, shuffle=True)
        validation_dl_r = data.DataLoader(validation_data_resized, b_value, shuffle=True)
        test_dl_r = data.DataLoader(test_data_resized, b_value, shuffle=True)

    # This function is used to train the current model for one epoch using the training data
    def train(self, model, optimizer, criterion):
        
        device = 'cuda' if cuda.is_available() else 'cpu'

        model.train()
        for data, target in enumerate(train_dl_r):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    # This function is used to validate the current model for one loop of the validation data
    def test(self, model, criterion):
        device = 'cuda' if cuda.is_available() else 'cpu'
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
         
    # This function will create an untrained object of the desired model. It will then take in the specified number of epochs and begin training and validating the model
    def start_training(self):
        device = 'cuda' if cuda.is_available() else 'cpu'
        if str(self.select_combo.currentText) == "EMNIST DNN":
            model = StandardNeuralNet()
        else:
            model = LeNet5NeuralNet()
        
        num_of_epochs = int(self.epoch_spin.value())
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        
        i = 1
        while i <= num_of_epochs:
    
            self.train(model, criterion, optimizer)
            self.test(model, criterion)
            i += 1
            
        
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Train_Button.setText(_translate("MainWindow", "Train Model"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.title.setText(_translate("MainWindow", "Handwritten Character Recognizer"))
        self.epoch_label.setText(_translate("MainWindow", "Number of Epochs"))
        self.batch_label.setText(_translate("MainWindow", "Batch Size"))
        self.train_cancel_button.setText(_translate("MainWindow", "Stop/Start"))
        self.download_label.setText(_translate("MainWindow", "Download"))
        self.train_label.setText(_translate("MainWindow", "Train"))
        self.train_ratio_label.setText(_translate("MainWindow", "Train"))
        self.validation_label.setText(_translate("MainWindow", "Validation"))
        self.accuracy_label.setText(_translate("MainWindow", "Accuracy:"))
        #self.select_combo.setItemText(0, _translate("MainWindow", "Lenet5 CNN"))
        #self.select_combo.setItemText(1, _translate("MainWindow", "EMNIST DNN"))
        self.Select_nn_label.setText(_translate("MainWindow", "Select NN:"))
        self.download_time_label.setText(_translate("MainWindow", "Est. Download Time:"))
        self.download_button.setText(_translate("MainWindow", "Download Dataset"))
        self.download_cancel_button.setText(_translate("MainWindow", "Stop/Start"))
        self.train_ratio.setText(_translate("MainWindow", "Train/Validation Ratio:"))
        self.actionTrain_Model.setText(_translate("MainWindow", "Train Model"))
        self.actionTrain_Model.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionQuite.setText(_translate("MainWindow", "Exit"))
        self.actionQuite.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.ratio_label.setText(_translate("MainWindow", "0.01"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.load_button.setText(_translate("MainWindow", "Load"))
        self.model_name_label.setText(_translate("MainWindow", "Model Name"))

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

class ThreadClass(QtCore.QThread):
	any_signal = QtCore.pyqtSignal(int)
	def __init__(self, parent=None,index=0):
		super(ThreadClass, self).__init__(parent)
		self.is_running = True

	def run(self):
		print('Starting thread...')
		cnt=0
		while cnt < 100:
			cnt+=1
			time.sleep(3)
			self.any_signal.emit(cnt)

	def stop(self):
		self.is_running = False
		print('Stopping thread...')
		self.terminate()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
