import os, glob, torch, cv2, pickle, json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .imagenet_models import ResNet18_Model


class ImageNetClassificationDataset(Dataset):
    '''
    The dataset class creates a cropped cell image with specified resolution with the tracked cell in center. 
    
    arguments:
        image_path      : A path where the images after preprocessing are stored
        track_path      : A path where the tracking results are stored
        file_extension  : The type of the file extension for the cell details are stored (pkl) 
        resolution      : The input resolution of the images to the model. it will be resized from the cropped dimension
        crop_width      : The width of the bounding box around cell to be cropped
        crop_height     : The heigh of the bounding box around cell to be cropped
    '''

    def __init__(self, image_path, track_path, file_extension = 'pkl', resolution= (224,224), crop_width= 96, crop_height = 96):
        
        self.imagepath = image_path
        self.resolution = resolution
        all_files =  glob.glob(os.path.join(track_path,'*.'+file_extension))

        #open all .pkl files and get the location and other deails about cell from each frame
        self.single_cells_info = []
        for file in all_files:
            with open(file, 'rb') as f:
                cell_data = pickle.load(f)
            filename = file.split('/')[-1].split('.')[0]
            for cell in cell_data.keys():
                (x,y) = cell_data[cell]['centroid']
                r = cell_data[cell]['radius']
                x1, y1 = x-int(crop_height/2), y-int(crop_width/2)
                x2, y2 = x+int(crop_height/2), y+int(crop_width/2)
                if x1 > 0 and x2 < 1024 and y1 > 0 and y2 < 1024: 
                    self.single_cells_info.append([filename, x1,x2, y1,y2, cell, x,y, r])

    def __len__(self):
        return len(self.single_cells_info)

    def __getitem__(self, index):
        data = self.single_cells_info[index]
        
        #load image data
        filepath = os.path.join(self.imagepath, data[0] + '.png')
        img1 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        #crop and resize
        img1 = img1[ data[3]:data[4], data[1]:data[2]]
        image = cv2.resize(img1, self.resolution )
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image = np.moveaxis(image, -1, 0)

        #to tensor and normalize
        image = torch.tensor(np.array(image, dtype= np.float32))
        image = image/torch.max(image)

        output = {'frame': data[0],'id': data[5], 'image': image, 'bb': [data[1],data[3],data[2],data[4]], 'centroid': [data[6],data[7]], 'radius' : data[8]}
        return output


class ImageNetClassification():
    '''
    class to handle the ResNet18 classification from the tracking results

    arguments:
        loghandler              : A loghandler object which can log the information into the logging file
        output_path             : The folder path where the output should be stored
        image_path              : The folder path to the original images folder
        track_path              : The folder where the tracking results are stored
        pretrained_model_path   : The filepath of pretrained model used for classification
        n_classes               : Number of classes to be predicted
        batch_size              : The batcj size to the model
        device_type             : Whether to do by GPU or CPU
    '''

    def __init__(self, loghandler, output_path, image_path, track_path, pretrained_model_path, n_classes, batch_size, device_type):
        self.loghandler = loghandler
        self.device_type = device_type
        self.output_path = output_path
        self.batch_size = batch_size
        self.imagepath = image_path

        self.dataset = ImageNetClassificationDataset(image_path = image_path, track_path= track_path, file_extension = 'pkl', resolution= (224,224), crop_width= 96, crop_height = 96)
        self.dataloader = DataLoader(self.dataset, batch_size= self.batch_size, shuffle= False)
        self.loghandler.log('info', str(len(self.dataloader)*self.batch_size) + " cells found for classification")
        self.model= ResNet18_Model(n_classes = n_classes)
        trained_model_s = torch.load(pretrained_model_path)
        self.model.load_state_dict(trained_model_s)
        self.model = self.model.to(torch.device(self.device_type))

        self.classification_result = []

    def classify(self):
        '''
        function to iterate through all images and predict the classes of each cell and then save the results to output folder
        '''
        with torch.no_grad():
            #=====Iteration over mini batches=====
            for didx,data in enumerate(self.dataloader):
                self.model.eval()

                images = data['image']
                frameids = data['frame']
                cell_ids = data['id']
                bbs = data['bb']
                centroids = data['centroid']
                radii = data['radius']

                images = images.to(torch.device(self.device_type))

                output, _ = self.model(images)
                labels = torch.argmax(output,dim=1)

                labels  = labels.cpu()

                for i, cell in enumerate(cell_ids):

                    self.classification_result.append( { 'image_path': os.path.join(self.imagepath, frameids[i] + '.png'),
                                                            'cell_id': cell_ids[i].item(),
                                                            'bb': (bbs[0][i].item(), bbs[1][i].item(), bbs[2][i].item(), bbs[3][i].item()),
                                                            'centroid': [centroids[0][i].item(), centroids[1][i].item()],
                                                            'radius': radii[i].item(),
                                                            'pred_class': labels[i].item()  } )


    def save_classification_results(self):
        '''
        function to save the results in a .json format to the output folder
        '''
        #save the results
        with open( os.path.join(self.output_path, 'classification_result.json' ), 'w') as fout:
                json.dump(self.classification_result, fout)


    def visualize_results(self):
        '''
        function to visualize the tracking results
        '''
        #visualize the results
        mitosis_stages = {0: 'interphase', 1: 'mitosis', 2: 'post-mitosis'}
        mitosis_color = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0)}
        grouped_dict = {}
        for my_dict in self.classification_result:
            image_path = my_dict['image_path']
            if image_path not in grouped_dict:
                grouped_dict[image_path] = []
            grouped_dict[image_path].append(my_dict)
        for image_path, grouped_dicts in grouped_dict.items():
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            filename = image_path.split('/')[-1]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for my_dict in grouped_dicts:
                cell_id = my_dict['cell_id']
                stage = my_dict['pred_class']
                centroid = my_dict['centroid']
                bb = my_dict['bb']
                rgb_image = cv2.putText(rgb_image, str(cell_id), centroid, cv2.FONT_HERSHEY_SIMPLEX,  0.5 , (51, 255, 255), 1, cv2.LINE_AA)
                rgb_image = cv2.rectangle(rgb_image, (bb[0]+25,bb[1]+25), (bb[2]-25,bb[3]-25) , mitosis_color[stage], 1)
                rgb_image =  cv2.putText(rgb_image , mitosis_stages[stage] , [bb[0]+23,bb[1]+23], cv2.FONT_HERSHEY_SIMPLEX, 0.4, mitosis_color[stage], 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.output_path, filename), rgb_image)


def build_model(args, loghandler, resultfolder, devicetype):
     
    image_path = os.path.join(resultfolder, 'images')
    suffix = '_postprocess' if args.is_tracking_postprocess else ''
    track_path = os.path.join(resultfolder, args.tracking_method + suffix)
    pretrained_model_path = args.pretrained_clf_model
    n_classes = 3
    batch_size = 16 
    device_type =  devicetype 

    output_path = os.path.join(resultfolder, 'classification_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    loghandler.log("info", "Starting ImageNet(ResNet18) based classification of cells...")
    clf = ImageNetClassification(loghandler, output_path, image_path, track_path, pretrained_model_path, n_classes, batch_size, device_type)
    clf.classify()
    clf.save_classification_results()
    clf.visualize_results()
    loghandler.log("info", "Completed ImageNet based classification of cells.  Saved results to folder classification_results")



    

