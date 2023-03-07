import os, glob, torch, cv2, csv, json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .imagenet_models import ResNet18_Model
from .gru_models import Base_Model, Time_Encoded_ResNet_Model

class SingleCellClassificationDataset(Dataset):
    """
    Dataset class to load the images and masks of single cells belonging to each sequence at a time.

    arguments:
        base_path:      Path to the folder containing all the single cell sequences
        image_format:   The images should be loaded in gray (96x96) or in RGB format (224x224) 
    """

    def __init__(self, base_path, image_format):
        
        self.base_path = base_path
        self.image_format = image_format
        self.load_image_paths()


    def load_image_paths(self):
        '''
        Function to load the names and path to all images
        '''
        self.all_frames = []
        for folder in os.listdir(self.base_path):    
            if "cell" in folder:
                #image paths
                path = os.path.join (self.base_path, folder, "Raw")      #inside each folder the images are under a folder named Raw
                files = []
                for file in os.listdir(path):
                    if ".png" in file:                       
                        files.append(os.path.join(path,file))           #[seqm_img1,..,seqm_imgN]
                files.sort()
            self.all_frames.append(files)
    

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, index):
        #all the frames belonging to a sequence
        sequence = self.all_frames[index]
        #read the name of the folder from the first image path.
        folder = sequence[0].split("/")[-3]

        images = []
        masks = []
        #iterate through each imagepath in a sequence in proper sorted order
        for path in sequence:
            #load image path 
            img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            #for rgb image change to the required dimensions
            if self.image_format == 'RGB':
                img = cv2.resize(img,(224,224))
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                img = np.moveaxis(img, -1, 0)
            else:
                img = img[None,:]
            #load mask path
            mask_path = path.replace("Raw","Mask")
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            mask[mask==255] = 1
            #for rgb image change to the required dimensions
            if self.image_format == 'RGB':
                mask = cv2.resize(mask,(224,224))
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
                mask = np.moveaxis(mask, -1, 0)
            else:
                mask = mask[None,:]
            #append the images and masks belong to same sequence to the lists
            masks.append(mask)
            images.append(img)

        #normalize images  masks and change to tensor
        images = np.array(images, dtype= np.float32)                       
        #Calcualting Lower and upper quantiles to normalize the image
        if self.image_format == 'RGB':
            images = images/ max(np.unique(images))
        else:
            #applying centre part quatile based normalization 
            Lq = np.quantile(images[:,:,42:54,42:54], 0.01)
            Uq = np.quantile(images[:,:,42:54,42:54], 0.99)                       
            images = (images - Lq)/ (Uq - Lq)
            images = np.clip(images,0,1)   
        images = torch.tensor(images, dtype=torch.float32)
        #normalize masks and change to tensor (as masks it is change to masked image)
        masks =np.array(masks, dtype= np.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        sample = { "image" : images , "mask" : masks, 'name': folder}
        return sample


class SingleCellSequencePrediction():
    """
    Class to load each sequence and predict the classes for each sequences depending on the given model

    arguments:
        loghandler              : A loghandler object which can log the information into the logging file
        base_path               : The base path where each tracklet sequences are saved
        image_format            : Whether to load images as gray 96x96 or RGB 224 x 224
        img_type                : Whether the batch is images or sequences
        batch_size              : Batch of sequences in prediction
        model_type              : Which is the model used for prediction
        pretrained_model_path   : Path to the pretrained model for the given model_type
        device_type             : cpu or gpu for training
        n_classes               : To be predicted into n number of classes (default n_classes = 3)
    """

    def __init__(self, loghandler, base_path, image_format, img_type, batch_size, model_type, pretrained_model_path, device_type, n_classes = 3):
        
        self.loghandler = loghandler
        self.base_path = base_path
        self.device_type = device_type
        self.img_type = img_type
        self.model_type = model_type

        self.dataset = SingleCellClassificationDataset(self.base_path, image_format)
        self.dataloader =  DataLoader(self.dataset, batch_size= batch_size, shuffle= True)
        if model_type == 'resnet18':
            self.model = ResNet18_Model(n_classes = n_classes)
        elif model_type == 'basemodel':
            self.model = Base_Model(n_classes = n_classes, num_layers=3, device = device_type)
        elif model_type == 'timeencodedresnet':
            self.model = Time_Encoded_ResNet_Model(n_classes = n_classes, num_layers=3, device = device_type)
        else:
            #error not implemented for other models
            assert 1==0

        trained_model_s = torch.load(pretrained_model_path)
        self.model.load_state_dict(trained_model_s)
        self.model = self.model.to(torch.device(self.device_type))



    def predict(self):
        '''
        function to go through each bath and predict the stages of mitosis
        '''
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):

                self.model.eval()
                images = data['image']      # shape: [batch_size, sequence, channels, height, width] 
                filenames = data['name']
                images = images.to(self.device_type)

                #prediction using models
                if self.img_type == 'img':
                    # reshape the tensor for the resnet prediction [batch_size*sequence, channels, height, width] 
                    batch_size, sequence, num_channels, height, width = images.shape
                    images = images.view(batch_size * sequence, num_channels , height, width)
                    output, _ = self.model(images)
                    labels = torch.argmax(output,dim=1)
                    labels =  labels.view(batch_size , sequence)
                else:
                    _, output, _ = self.model(images)
                    labels = torch.argmax(output,dim=2)
                #assign to cpu numpy
                labels = labels.cpu().detach().numpy()
                #save results
                for i,label_seq in enumerate(labels):
                    self.save_results(label_seq, filenames[i])


    def save_results(self, labels, filename):
        '''
        function to save the results for each seuence into the folder
        '''
        labels = labels.astype(np.int32)
        filepath = os.path.join(self.base_path, filename, self.model_type + '_predicted.txt')
        np.savetxt(filepath, labels, delimiter=',', fmt='%d')    
        self.loghandler.log("info", "Saved results for " + filename + " : " + str(labels))



    def single_cell_classification_visualization_wholeslde(self, imagepath, outputpath):
        '''
        Function to visualize the extracted tracklets are corresponding classes on full images
                results will be stored in outputpath    
        '''
        basepath = self.base_path
        cell_data = []
        list_images = []
        mitosis_stages = {0: 'interphase', 1: 'mitosis', 2: 'post-mitosis'}
        mitosis_color = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0)}
        for folder in os.listdir(basepath):
            #get prediction results
            predictions = []
            pred_filepath = os.path.join(basepath, folder, self.model_type + '_predicted.txt')
            with open(pred_filepath, 'r') as txtfile:
                for line in txtfile:
                    predictions.append(int(line))

            #get cell crop details
            filepath  = os.path.join(basepath,folder,'crop_details.csv')
            with open(filepath) as csvfile:
                reader = csv.reader(csvfile)
                for i,row in enumerate(reader):
                    cell_data.append([row[6].split('/')[-1], int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), predictions[i] ])  
                    list_images.append( row[6].split('/')[-1] )

        #now go through each image and plot the predictions
        for imagename in os.listdir(imagepath):
            if '.png' in imagename:
                image = cv2.imread(os.path.join(imagepath, imagename), cv2.IMREAD_UNCHANGED)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                if imagename in list_images:
                    for data in cell_data:
                        if data[0] == imagename:
                            bb = [data[3], data[4], data[5], data[6]]
                            stage = data[-1]
                            cell_id = data[2]
                            centroid = [bb[0]+48, bb[1]+ 48]
                            rgb_image = cv2.putText(rgb_image, str(cell_id), centroid, cv2.FONT_HERSHEY_SIMPLEX,  0.5 , (51, 255, 255), 1, cv2.LINE_AA)
                            rgb_image = cv2.rectangle(rgb_image, (bb[0]+25,bb[1]+25), (bb[2]-25,bb[3]-25) , mitosis_color[stage], 1)
                            rgb_image =  cv2.putText(rgb_image , mitosis_stages[stage] , [bb[0]+23,bb[1]+23], cv2.FONT_HERSHEY_SIMPLEX, 0.4, mitosis_color[stage], 1, cv2.LINE_AA)

                cv2.imwrite(os.path.join(outputpath, imagename), rgb_image)





def build_model(args, loghandler, resultfolder, devicetype):

    #defining the input image types for the models
    if args.single_cell_classification_method == 'resnet18':
        img_type = 'img'
    else:
        img_type = 'seq'
    if args.single_cell_classification_method == 'basemodel' :
        image_format = 'gray'           
    else:
        image_format = 'RGB'  
    #batch size includes number of sequences
    batch_size = 2
    n_classes = 3

    base_path = os.path.join(resultfolder, 'single_cells')
    model_type = args.single_cell_classification_method
    pretrained_model_path = args.pretrained_single_cell_clf_model

    loghandler.log("info", "Starting " + model_type + " based classification of single cells tracklets...")
    s = SingleCellSequencePrediction(loghandler, base_path, image_format, img_type, batch_size, model_type, pretrained_model_path, devicetype, n_classes)
    s.predict()
    loghandler.log("info", "Completed classification of cells tracklets.  Saved results to each subfolder in folder single_cells")

    image_path = os.path.join(resultfolder, 'images')
    visualize_path = os.path.join(resultfolder, 'classification_tracklets_visualized')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    s.single_cell_classification_visualization_wholeslde(image_path, visualize_path)
    loghandler.log("info", "Visualized classification of cells tracklets in folder classification_tracklets_visualized")

