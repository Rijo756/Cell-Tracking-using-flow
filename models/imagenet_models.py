import numpy as np
import torch, cv2
from torch import nn 
import torchvision


class ResNet18_Model(nn.Module):

    def __init__(self, n_classes = 6):
        '''
        Initialization for the recurrent tracking
        
        Parameters:
            n_classes: (int)
                Number of classes for the classification of labels.
        '''
        
        super(ResNet18_Model, self).__init__()

        self.n_classes  = n_classes

        #load a prelaoded ResNet18Model
        self.embeddingnet = torchvision.models.resnet18(pretrained=True)

        #last Fully connected layer output to n_classes
        self.linear = nn.Sequential(
                                    nn.Linear(1000, self.n_classes)
                                    )
                                    
    def forward(self, inputs):
        '''
        Forward Propagation of the ResNet18 model architecture
        
        Parameters:
            inputs : (tensor --> [batchsize, input channels, width, height])

        Return:
            states : (tensor --> [batchsize, n_classes])  #classification network output
            embedding: (tensor --> [batchsize, hidden channels, width, height]) #embedding space from the network
        '''

        embeddings = self.embeddingnet(inputs)
        states = self.linear(embeddings)

        return states, embeddings


class DenseNet121_Model(nn.Module):

    def __init__(self, n_classes = 6):
        '''
        Initialization for the recurrent tracking
        
        Parameters:
            n_classes: (int)
                Number of classes for the classification of labels.
        '''
        
        super(DenseNet121_Model, self).__init__()

        self.n_classes  = n_classes

        #load a prelaoded DenseNet121 Model
        self.embeddingnet = torchvision.models.densenet121(pretrained = True , drop_rate= 0 )

        #last Fully connected layer output to n_classes
        self.linear = nn.Sequential(
                                    nn.Linear(1000, self.n_classes)
                                    )
                                    
    def forward(self, inputs):
        '''
        Forward Propagation of the DenseNet121 model architecture
        
        Parameters:
            inputs : (tensor --> [batchsize, input channels, width, height])

        Return:
            states : (tensor --> [batchsize, n_classes])  #classification network output
            embedding: (tensor --> [batchsize, hidden channels, width, height]) #embedding space from the network
        '''

        embeddings = self.embeddingnet(inputs)
        states = self.linear(embeddings)

        return states, embeddings

