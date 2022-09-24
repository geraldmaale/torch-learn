from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, output_size)
        
    def forward(self, x:torch.Tensor):
        z = self.l1(x)
        z = self.l2(z)
        z = self.l3(z)
        z = self.l4(z)
        return z


class DeepNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepNet, self).__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, output_size)
        
    def forward(self, x:torch.Tensor):
        z = torch.relu(self.l1(x))
        z = torch.relu(self.l2(z))
        z = torch.relu(self.l3(z))
        z = self.l4(z)
        return z
    
    
class DeepNetV2(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepNetV2, self).__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, output_size)
        
    def forward(self, x:torch.Tensor):
        z = torch.tanh(self.l1(x))
        z = torch.tanh(self.l2(z))
        z = torch.tanh(self.l3(z))
        out = self.l4(z)        
        return out
    
    
class LeNet(torch.nn.Module):
    def __init__(self, input_shape, output_size, batch_size):
        super(LeNet, self).__init__()
        self.batch_size = batch_size
        self.num_classes = output_size
        self.momentum = 0.9
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(3, 64, 3) # 32 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(64, 128, 3) # (32, 32, 32)
        self.conv3 = nn.Conv2d(128, 256, 3) # (128, 128, 3)
        self.pool = nn.MaxPool2d(3, 3) # kernel_size, stride

        n_size = self._get_conv_output()

        self.fc1 = nn.Linear(n_size, 256) # Dense layer
        self.fc2 = nn.Linear(256, 128) # Dense layer
        self.fc3 = nn.Linear(128, self.num_classes) # Dense layer
        self.flat = nn.Flatten() # Flatten layer
        self.dropout = nn.Dropout(0.4) # Dropout layer
        self.softmax = nn.Softmax(dim=1) # Softmax layer

    def _get_conv_output(self):
        input = torch.autograd.Variable(torch.rand(self.batch_size, *self.input_shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(self.batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x:torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor):
        x = self._forward_features(x)
        x = self.flat(x)  
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
class FMnistModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super(FMnistModelV0, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),      
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
        
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)
    
    
class FMnistModelV1(nn.Module):
    def __init__(self, 
                 input_shape:int, 
                 hidden_units:int, 
                 output_shape:int):
        super(FMnistModelV1, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(in_features=input_shape,
                            out_features=hidden_units)
        self.l2 = nn.Linear(in_features=hidden_units,
                            out_features=output_shape)
        
    def forward(self, x:torch.Tensor):        
        return (self.l2(self.l1(self.flatten(x))))
    

class FMnistModelV2(nn.Module):
    """Fully connected CNN model with 2 hidden layers"""
    def __init__(self, 
                 input_shape:torch.Size, 
                 hidden_units:int, 
                 output_shape:int,
                 batch_size:int):
        super(FMnistModelV2, self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        
        # get conv output
        n_size = self._get_conv_output()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= n_size, # conv output size as input
                      out_features=output_shape)
        )
        
    def _get_conv_output(self):
        input = torch.autograd.Variable(torch.rand(self.batch_size, *self.input_shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(self.batch_size, -1).size(1)
        return n_size
    
    def _forward_features(self, x:torch.Tensor):
        x = self.conv_block_2(self.conv_block_1(x))
        return x
        
    def forward(self, x:torch.Tensor):  
        # size = self._get_conv_output()      
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        out = self.classifier(x)
        # print(out.shape)
        return out
    

class FMnistModelCNNV2(nn.Module):
    """Fully connected CNN model with 2 hidden layers"""
    def __init__(self, 
                 input_shape:torch.Size, 
                 hidden_units:int, 
                 output_shape:int):
        super(FMnistModelCNNV2, self).__init__()
        self.input_shape = input_shape
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, # inspect the shape of the block before the classifier (8, 8)
                      out_features=output_shape)
        )
        
    def _get_conv_output(self):
        input = torch.autograd.Variable(torch.rand(self.input_shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(self.batch_size, -1).size(1)
        print(f"n_size: {n_size}")
        return n_size
    
    def _forward_features(self, x:torch.Tensor):
        out = self.conv_block_1(self.conv_block_2(x))
        return out
        
    def forward(self, x:torch.Tensor):  
        size = self._get_conv_output()      
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        out = self.classifier(x)
        # print(out.shape)
        return out