import torch, glob, os, cv2, pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from .unet_architecture import UNetArchitecture


class SegmentationDataset(Dataset):
    '''
    Dataset class to read and return image and image name from the given folder and with extensions.

    arguments:
        base_folder:    The folder path where all the images are stored
        file_extension: Format of the image files
    '''

    def __init__(self, base_folder = None, file_extension = 'png'):
        super(SegmentationDataset, self).__init__()
        self.all_frames =  glob.glob(os.path.join(base_folder,'*.'+file_extension))
        self.all_frames.sort()

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        imagepath = self.all_frames[idx]
        filename = imagepath.split('/')[-1]
        img1 = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
        img1 = np.array(img1, np.uint8)
        if len(img1.shape) == 2:  #add additional channel if gray scale
            img1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))
        img1 = img1/torch.max(img1)

        return {'image': img1, 'name': filename}
        

class UNet_Watershed():
    """
    To do the UNet based semantic segmentation and apply watershed algorithm to find the instance segmentaion.

    arguments:
        loghandler:                 A loghandler object which can log the information into the logging file
        input_path:                 The folder path to the images folder
        file_type:                  Format of the images stored
        output_path:                The file path to store the images
        batch_size:                 Batch size of the model to do the segmentation
        use_pretrained_model:       Whether to use a pretrained model
        pretrained_model_path:      The file path of the model saved for the segmentation using UNet
        device_type:                Which device to use for training either GPU or CPU
        input_channels:             The number of channels input to the UNet (1 channel - only one image input)
        depth:                      The depth of the UNet architecure (default: 3)
        is_eliminate_boundarycells: Whether to eliminate all the cell at boundary
        is_eliminate_smallcells:    Whether to eliminate small cells
        is_eliminate_largecells:    Whether to eliminate large cells
        is_eliminate_lessdensecells:Whether to eliminate less dense cells
    """

    def __init__(self, loghandler, input_path, file_type, output_path, batch_size, use_pretrained_model, pretrained_model_path, device_type, input_channels = 1, depth = 3,
                                    is_eliminate_boundarycells = False, is_eliminate_smallcells = True, is_eliminate_largecells = False, is_eliminate_lessdensecells = False):
    
        self.model = UNetArchitecture(num_classes = 1, in_channels= input_channels , depth= depth)
        self.loghandler = loghandler
        self.device_type = device_type

        if use_pretrained_model:
            trained_model_s = torch.load(pretrained_model_path)
            self.model.load_state_dict(trained_model_s)
        self.model = self.model.to(torch.device(self.device_type))

        self.inputpath = input_path
        self.output_path = output_path
        self.batch_size = batch_size

        self.dataset = SegmentationDataset(base_folder = self.inputpath, file_extension = file_type)
        self.dataloader = DataLoader(self.dataset, batch_size= self.batch_size, shuffle= False)
        self.loghandler.log("info", str(len(self.dataloader)*self.batch_size) +" images found")

        #conditions for watershed instance segmentaions
        self.is_eliminate_boundarycells = is_eliminate_boundarycells
        self.is_eliminate_smallcells = is_eliminate_smallcells
        self.is_eliminate_largecells = is_eliminate_largecells
        self.is_eliminate_lessdensecells = is_eliminate_lessdensecells

    def test_time_augmentaion(self, tensor_input):
        '''
        Function to do the test time augmentation and average the predictions. Implemented flip and rotation
        '''
        #org image
        pred1 = self.model(tensor_input)
        
        #flip
        tensor_flip =  torch.flip(tensor_input,dims=[2])
        pred2 = self.model(tensor_flip)
        pred2 = torch.flip(pred2,dims=[2])

        #rotation 90
        tensor_rot90 = torch.rot90(tensor_input,k=1,dims=[2,3])
        pred3 = self.model(tensor_rot90)
        pred3 = torch.rot90(pred3,k=3,dims=[2,3])

        #rotation 270
        tensor_rot270 = torch.rot90(tensor_input,k=3,dims=[2,3])
        pred4 = self.model(tensor_rot270)
        pred4 = torch.rot90(pred4,k=1,dims=[2,3])

        return (pred1 + pred2 + pred3 + pred4)/4 


    def watershed_algorithm(self, pred_mask, orig_image):
        '''
        Apply watershed algorithm over the segmentation results to seperate the cell flows.
        '''
        #thresholding
        _, thresh = cv2.threshold(pred_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        #distance map and local maximas    
        thresh = opening
        D = ndi.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=25,labels=thresh)

        #watershed
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        #Get their cell inforamtions from watershed results for each cell
        cell_data = {}
        unq = np.delete(np.unique(labels),0)
        for cell in unq:
            region = np.where(labels == cell)
            cellregion=np.zeros(labels.shape)
            cellregion[labels == cell]=1

            #erode get a smaller segment
            cellregion = cv2.morphologyEx(cellregion, cv2.MORPH_ERODE, kernel, iterations=1)
            labels[(labels==cell) & (cellregion!=1)] = 0

            #calcualte cell density of each cell to eliminate cells
            cell_density = int(np.sum(np.multiply(orig_image, cellregion)))
            
            #bounding box, radius of cell inside bounding box and centroid of each cell
            bb = np.array([np.min(region[1]), np.min(region[0]), np.max(region[1]), np.max(region[0])]) 
            radius = int(np.max([bb[2] - bb[0], bb[3] - bb[1]]) / 2)
            centroid = (int(region[1].mean()), int(region[0].mean()))
            
            #to eliminate smaller cells
            if radius <= 8 and self.is_eliminate_smallcells:
                labels[labels==cell] = 0
                continue
            
            #check condition for borders and eliminate
            if self.is_eliminate_boundarycells:
                H,W = pred_mask.shape
                if centroid[0]-radius < -1 or centroid[0]+radius > W or centroid[1]-radius < -1 or centroid[1]+radius > H:
                    labels[labels==cell] = 0
                    continue

            #eliminate cells with less cell density
            if cell_density < 60 and self.is_eliminate_lessdensecells:
                labels[labels==cell] = 0
                continue

            #to eliminate bigger cells
            if radius > 45 and self.is_eliminate_largecells:
                labels[labels==cell] = 0
                continue
            
            #cell details with all the informations
            cell_data[cell] = {'bounding_box': bb, 'centroid': centroid, 'radius': radius, 'cell_density': cell_density}

        return labels, cell_data


    def instance_segmentation_prediction(self):
        '''
        for each image in the dataset do the UNet based semantic segmentation and call watershed alroithm to find instance segmentation
        '''
        with torch.no_grad():
            #=====Iteration over mini batches=====
            for didx,data in enumerate(self.dataloader):
                self.model.eval()

                image =  data['image']
                image = image.to(torch.device(self.device_type))
                name = data['name']
                #apply test time augmentation
                image_seg_pred = self.test_time_augmentaion(image)

                #post processing and save image
                for i in range(image_seg_pred.shape[0]):
                    pred = image_seg_pred[i][0].cpu().detach().numpy()
                    _, output = cv2.threshold(pred*255, 2, 255, cv2.THRESH_BINARY)
    
                    output = np.array(output,dtype=np.uint8)
                    img = image[i][0].cpu().detach().numpy()*255
                    img = np.array(img,dtype=np.uint8)
                    #watershed algorithm to find instance segmentation from semantic segmentation
                    labels, cell_data = self.watershed_algorithm(output, img)

                    filename = name[i]
                    self.loghandler.log('info', "\t Instance segmentation image: " + str(filename))
                    cv2.imwrite(os.path.join(self.output_path, filename), labels)
                    with open(os.path.join(self.output_path, filename.split('.')[0] + '.pkl'), 'wb') as f:
                        pickle.dump(cell_data, f)


def build_model(args, loghandler, resultfolder, devicetype):
    '''
    This function sets all the values and calls the objects accordingly.
    '''
    inputpath = os.path.join(resultfolder,'images')
    outputpath = os.path.join(resultfolder,args.inst_segmentation)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    filetype = 'png'
    batchsize = 2
    pretrainedmodel = args.pretrained_unetsegmentaion_model
    loghandler.log("info", "Starting the image UNet based segmentaion...")
    useg = UNet_Watershed(loghandler, input_path = inputpath, file_type = filetype, output_path = outputpath, batch_size = batchsize,
                                     use_pretrained_model = True, pretrained_model_path =pretrainedmodel, device_type = devicetype)
    useg.instance_segmentation_prediction()
    loghandler.log("info", "Completed image UNet based instance segmentaion. Saved results to folder "+ args.inst_segmentation )
