import torch  
import matplotlib.pyplot as plt  
import numpy as np  
import torch.nn.functional as func  
import PIL.ImageOps  
from torch import nn  
from torchvision import datasets,transforms   
import requests  
from PIL import Image  

transform1=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])  

training_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform1)  

validation_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transform1)  

training_loader=torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100,shuffle=True) 

validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=100,shuffle=False) 

def im_convert(tensor):  
    image=tensor.clone().detach().numpy()  
    image=image.transpose(1,2,0)  
    print(image.shape)  
    image=image*(np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5)))  
    image=image.clip(0,1)  
    return image  

dataiter=iter(training_loader)  
images,labels=dataiter.next()  
fig=plt.figure(figsize=(25,4))  

for idx in np.arange(20):  
    ax=fig.add_subplot(2,10,idx+1)  
    plt.imshow(im_convert(images[idx]))  
    ax.set_title([labels[idx].item()])  
    
#building the engine  Defining the layers of neural network
class classification1(nn.Module):  
    def __init__(self,input_layer,hidden_layer1,hidden_layer2,hidden_layer3, output_layer):  
        super().__init__()  
        self.linear1=nn.Linear(input_layer,hidden_layer1)  
        self.linear2=nn.Linear(hidden_layer1,hidden_layer2)  
        self.linear3=nn.Linear(hidden_layer2,hidden_layer3) 
        self.linear4=nn.Linear(hidden_layer3,output_layer)
        
    def forward(self,x):  
        x=func.relu(self.linear1(x))  
        x=func.relu(self.linear2(x))  
        x=self.linear3(x)  
        return x  
    
#defining the parametrs   
model=classification1(784,500,125,65,10)

#loss function
criteron=nn.CrossEntropyLoss()  

#optimization algorithm
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)  

epochs=12

loss_history=[]  

correct_history=[]  

val_loss_history=[]  
val_correct_history=[]

for e in range(epochs):  
    loss=0.0  
    correct=0.0  
    val_loss=0.0  
    val_correct=0.0  

    
    for input,labels in training_loader:  
        inputs=input.view(input.shape[0],-1)  
        outputs=model(inputs)  
        loss1=criteron(outputs,labels)  
        optimizer.zero_grad() 
        
        loss1.backward() 
        
        optimizer.step()  
        _,preds=torch.max(outputs,1)  
        loss+=loss1.item()  
        correct+=torch.sum(preds==labels.data)  
        
    else:  
        with torch.no_grad():  
            for val_input,val_labels in validation_loader:  
                val_inputs=val_input.view(val_input.shape[0],-1)  
                val_outputs=model(val_inputs)  
                val_loss1=criteron(val_outputs,val_labels)   
                _,val_preds=torch.max(val_outputs,1)  
                val_loss+=val_loss1.item()  
                val_correct+=torch.sum(val_preds==val_labels.data)  
                
        epoch_loss=loss/len(training_loader.dataset)  
        epoch_acc=correct.float()/len(training_dataset)  
        loss_history.append(epoch_loss)  
        correct_history.append(epoch_acc)  
          
        val_epoch_loss=val_loss/len(validation_loader.dataset)  
        val_epoch_acc=val_correct.float()/len(validation_dataset)  
        val_loss_history.append(val_epoch_loss)  
        val_correct_history.append(val_epoch_acc)  
        print('training_loss:{:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item()))  
        print('validation_loss:{:.4f},{:.4f}'.format(val_epoch_loss,val_epoch_acc.item()))  

url='https://cdn.discordapp.com/attachments/440010605752221696/969154870907142164/IMG_0318.jpg'
response=requests.get(url,stream=True)  
img=Image.open(response.raw)  
img=PIL.ImageOps.invert(img)  
img=img.convert('1')  
img=transform1(img)   
plt.imshow(im_convert(img))  

img=img.view(img.shape[0],-1)  
output=model(img)  
_,pred=torch.max(output,1)  
print(pred.item()) 