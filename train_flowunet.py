from matplotlib import pyplot as plt
import os, glob
import numpy as np
import torch, cv2
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from utils.LogHandler import LogHandler
from utils.Data import FlowDataset
from models.unet_architecture import UNetArchitecture

from utils.Augmentations import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomBrightness, RandomContrast, RandomSaturation, RandomHue, RandomGamma, RandomGaussianBlur, RandomSharpness


#defining arguments for training
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--image_path',type=str, default="/home/students/roy/trackingcells/images",help="Path of the input data file")
parser.add_argument('--subfolder_pattern',type=dict, default={"03":"03_GT/TRA","04":"04_GT/TRA"},help="Dictionary contain image and mask folder path/name")
parser.add_argument('--out_path',type=str, default="Script Path / Train_Flow",help="Will generate during the run")

parser.add_argument('--batch_size',type=int, default= 2, help="Size of the batch, for deeptracking it is 1")
parser.add_argument('--lr',type=float, default=0.001, help="learning rate")
parser.add_argument('--iter',type=int, default= 101 , help="Number of epochs")
parser.add_argument('--steps',type=int, default=0, help="Count the steos during training")

parser.add_argument('--use_pretrained_model',type=bool, default=False, help="Whether to use the pretrained model to start the training")
#parser.add_argument('--saved_model_path',type=str, default="/home/students/roy/Rijo_hiwi/tracking_network_trained_20_mask.pth",help="Path of the saved model")
args = parser.parse_args()


def plot_loss(args, epochs, trainloss, validationloss, label1 ='Training loss', label2 = 'validation loss', name = 'training_progress'):
    """
    Plotting the loss functions
    arguments:
        args             : system arguments
        epochs           : a list of epoch values to be plotted (x axis)
        trainloss        : a list of the loss occured during training for the epochs
        validationloos   : a list of the loss occcured during validations
    """
    #===== Plotting the losses per epoch to see learning=====
    plt.figure(1)
    plt.plot(epochs, trainloss, 'green', label=label1)
    plt.plot(epochs, validationloss, 'red', label=label2)
    plt.title('Training and Validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if args.steps <= 1:
        plt.legend()
    plt.savefig(os.path.join(args.out_path,"UnetfLowdetector_"+ name +".png"))



if __name__ == "__main__":
    #using GPU if available
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    #creating a Log handler for logging the progress
    script_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_path,"Train_Flow")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    loghandler = LogHandler(name = "Rijo", basepath=out_path)
    args.out_path = out_path

    #Adding the augmentation to the dataset
    transforms = Compose([ RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomRotation([90.0, 90.0]),    
                            RandomRotation([270.0, 270.0]), 
                            RandomRotation([180.0, 180.0]),
                            RandomBrightness(),
                            RandomContrast(), 
                            #RandomSaturation(),
                            #RandomHue(), 
                            #RandomGamma(), 
                            #RandomGaussianBlur(), 
                            #RandomSharpness()
                                        ])


    dataset = FlowDataset(base_folder = args.image_path, subfolder_pattern = args.subfolder_pattern , loghandler = loghandler, time_skips = [1,2,3], transforms = transforms )
    dataloader = DataLoader(dataset, batch_size= args.batch_size, shuffle= True)

    loghandler.log("info", "training data count: " + str(len(dataset)))
    loghandler.log("info", "batch size: " + str(args.batch_size))
    loghandler.log("info", "training batch count: " + str(len(dataloader)))

    #=====Model Network for training=====
    model = UNetArchitecture(num_classes = 1, in_channels= 2 , depth= 3)

    if args.use_pretrained_model:
        trained_model = torch.load(args.saved_model_path)
        model.load_state_dict(trained_model)
    model = model.to(torch.device(device_type))

    criterion1 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= 1e-5)

    trainloss = []
    epochs = []
    args.steps = 0

    #=====Iteration over Epochs=====
    for iter in range(args.iter):

        total_loss = 0
        model.train()
        loghandler.log("info", "Epoch " + str(iter+1) + ' of ' + str(args.iter))
        #=====Iteration over mini batches=====
        for didx,data in enumerate(dataloader):

            args.steps += 1

            inputdata = data['images']
            inputdata = inputdata.to(torch.device(device_type))

            target = data['flow']
            target = target.to(torch.device(device_type))

            optimizer.zero_grad()
            
            #=====Forward=====
            output = model(inputdata)
    
            #=====Backward=====
            loss =  criterion1(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() 
            loghandler.log("info", "\tbatch: " + str(didx+1) + '/' + str(len(dataloader)) + ' loss1:' + str(loss.item()) )

        loghandler.log("info", "Training Loss on Epoch "+str(iter + 1)+ " : "+str(total_loss))
        trainloss.append(total_loss)
        epochs.append(iter+1)
        plot_loss(args, epochs, trainloss, trainloss)

        if iter % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_path,"flow_network_trained_Hela_aug_"+str(iter)+".pth") )
            loghandler.log('info', "Model Saved in ../flow_network_trained_Hela_aug_"+str(iter)+".pth ")


