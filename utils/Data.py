import numpy as np
import torch, os, glob, math
import cv2
from .LogHandler import LogHandler
from torch.utils.data import Dataset
import numpy as np

class FlowDataset(Dataset):


    def __init__(self,  base_folder = None, subfolder_pattern = {} , loghandler = None, search_string='*tif', time_skips = [], is_test = False, transforms = None ):

        super(FlowDataset, self).__init__()

        self.slides = {}
        self.frames = []
        self.subfolder_pattern = subfolder_pattern
        self.is_test = is_test

        self.transforms =  transforms 

        #creating the log handler
        if loghandler is not None:
             self.loghandler = loghandler
        else:
            script_path = os.path.dirname(os.path.realpath(__file__))
            out_path = os.path.join(script_path,"output")
            self.loghandler = LogHandler(name = "Rijo", basepath=out_path)

        #read the names of the input image files from base folder.
        self.loghandler.log("info", "Input base folder: " + str(base_folder))

        for folder in subfolder_pattern.keys():
            input_path = os.path.join(base_folder, folder)
            self.loghandler.log("info", "Images load from folder: " + str(input_path))
            if isinstance(search_string, list):
                for search_s in search_string:
                    all_frames =  glob.glob(os.path.join(input_path,search_s))
                    all_frames.sort()
                    if len(all_frames) == 0:
                        continue
                    self.slides[folder] = all_frames
            else:
                all_frames =  glob.glob(os.path.join(input_path,search_string))
                all_frames.sort()
                if len(all_frames) == 0:
                        continue
                self.slides[folder] = all_frames
                
        for ts in [0] + time_skips:
            for folder in self.slides.keys():
                slides = self.slides[folder]
                for s_ind in range(len(slides)-ts - 1):
                    self.frames.append([slides[s_ind], slides[s_ind + ts + 1], folder])

        assert self.slides is not None

        #print (self.frames)

        self.sequences = len(self.slides)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        #input image loading
        imgpath1 = self.frames[idx][0]
        imgpath2 = self.frames[idx][1]
        img1 = cv2.imread(imgpath1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(imgpath2, cv2.IMREAD_UNCHANGED)
        #print (imgpath1, imgpath2)
        img1 = (img1 - np.min(img1))*65655/(np.max(img1)-np.min(img1))
        img2 = (img2 - np.min(img2))*65655/(np.max(img2)-np.min(img2))

        #mask images loading with folder name
        key = self.frames[idx][2]
        maskpath1 = imgpath1.replace('/'+key + '/t' ,'/' + self.subfolder_pattern[key]  + '/man_track')
        maskpath2 = imgpath2.replace('/'+key + '/t' ,'/' + self.subfolder_pattern[key]  + '/man_track')
        mask1 = cv2.imread(maskpath1, cv2.IMREAD_UNCHANGED)
        mask2 = cv2.imread(maskpath2, cv2.IMREAD_UNCHANGED)

        #loading tracking info  
        track_file_path,_ = os.path.split(os.path.abspath(maskpath1))
        track_file_path = os.path.join(track_file_path,'man_track.txt')
        parent_cells = self.find_parent_cells(track_file_path)

        #convert to tensor for training
        #inp_image = torch.tensor(np.array(inp_image, dtype= np.float32))
        image1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))
        image2 = torch.tensor(np.array(img2[None,:], dtype= np.float32))
        seg_mask1 = torch.tensor(np.array(mask1[None,:], dtype= np.float32))
        seg_mask2 = torch.tensor(np.array(mask2[None,:], dtype= np.float32))
        #print (np.unique(mask1), print (torch.unique(seg_mask1)))

        #applying same transfoms
        if self.transforms != None:
            image1, image2, seg_mask1, seg_mask2 = self.transforms(image1, image2, seg_mask1, seg_mask2 )

        #find the flow and track images
        flow_image, track_image = self.find_trackflow(seg_mask1[0].cpu().detach().numpy(), seg_mask2[0].cpu().detach().numpy(), parent_cells)
        flow_image = torch.tensor(np.array(flow_image[None,:], dtype= np.float32))

        #thresholding masks
        seg_mask1[seg_mask1 > 0] = 255
        seg_mask2[seg_mask2 > 0] = 255

        #stack images for training
        inp_image = torch.cat((image1,image2),  0)
            
        #normalizing the data to range 0 to 1
        inp_image = inp_image/torch.max(inp_image)
        image1 = image1/torch.max(image1)
        image2 = image2/torch.max(image2)
        seg_mask1 = seg_mask1/torch.max(seg_mask1)
        seg_mask2 = seg_mask2/torch.max(seg_mask2)
        flow_image = flow_image/torch.max(flow_image)

        #dataset return data
        output = {'images': inp_image, 'image1' : image1, 'image2': image2, 'mask1': seg_mask1, 'mask2': seg_mask2,  'flow': flow_image}

        return output



    def find_cell_details(self, seg_mask):
        '''
        Getting the track segmentation mask this funtion will find the details of each cell like bounding box, centroid and radius
        '''
        cell_details = {}
        #find cell details for each unique cell
        for cell in np.unique(seg_mask):
            if cell == 0:
                continue
            region = np.where(seg_mask == cell)
            cellregion=np.zeros(seg_mask.shape)
            cellregion[seg_mask == cell]=1
            bb = [np.min(region[1]), np.min(region[0]), np.max(region[1]), np.max(region[0])]    #  (x,y)
            radius = int(np.max([bb[2] - bb[0], bb[3] - bb[1]]) / 2)
            centroid = (int(region[1].mean()), int(region[0].mean()))       # (x,y) / (column, row)
            cell_details[cell] = {'bb': bb, 'centroid': centroid, 'radius': radius}

        return cell_details

    
    def find_trackflow(self, seg_mask1, seg_mask2, parent_cells):
        '''
        Getting the segmentation masks of two frames and parent cell information this will 
        1. track image : create a line from centroid of a cell in one frame to other. 
        2. flow image  : create a cell flow track for a cell in one frame to other. 
        '''
        #get cell details
        cell_details1 = self.find_cell_details(seg_mask1)
        cell_details2 = self.find_cell_details(seg_mask2)

        #get unique track ids
        unq_img1 = np.delete(np.unique(seg_mask1),0)
        unq_img2 = np.delete(np.unique(seg_mask2),0)
        #print (unq_img1, unq_img2)
        #create two dark images for track and flow
        flow_image = np.zeros(seg_mask1.shape)
        track_image = np.zeros(seg_mask1.shape)

        #Go through each unique ids in both frames 
        for el1 in unq_img2:
            if el1 in unq_img1:
                el2 = el1
            else:
                if parent_cells[el1] in unq_img1:
                    el2 = parent_cells[el1]
                else:
                    #print (el1, 'not found')
                    continue
            
            # get details of the cell in both frame
            frame1_details =  cell_details1[el2]
            frame2_details =  cell_details2[el1]

            #draw a line between centroids
            track_image =  cv2.line(track_image, frame1_details['centroid'], frame2_details['centroid'], 255, 2)

            #to draw the flow 
            x1 = frame1_details['centroid'][0]
            y1 = frame1_details['centroid'][1]
            x2 = frame2_details['centroid'][0]
            y2 = frame2_details['centroid'][1]
            #drawing the flow with the radius found of both cells
            slope = (y2-y1)/(x2-x1+1e-5)
            dist = frame1_details['radius'] - 5
            dy = int(math.sqrt(dist**2/(slope**2 + 1)))
            dx = int(-slope * dy)
            p1 = [x1+dx,y1+dy]
            p2 = [x1-dx,y1-dy]
            dist = frame2_details['radius'] - 5
            dy = int(math.sqrt(dist**2/(slope**2 + 1)))
            dx = int(-slope * dy)
            p3 = [x2+dx,y2+dy]
            p4 = [x2-dx,y2-dy]
            #create a contour
            ctr = [np.array([p1,p2,p4,p3], dtype=np.int32)]
            #draw the flows including the both the cells
            flow_image[seg_mask1 == el2] = 255
            flow_image = cv2.drawContours(flow_image, ctr ,0,255,-1)
            flow_image[seg_mask2 == el1] = 255
                
        return flow_image, track_image



    def find_parent_cells(self, track_file_path):
        parent_cells = {}
        with open(track_file_path,'r') as file:
            lines = file.read().splitlines()
        track  = [ list(map(int, l.split())) for l in lines]
        for tr in track:
            parent_cells[tr[0]] = tr[3]
        
        return parent_cells


class TestImageDataset(Dataset):

    def __init__(self,  base_folder = None ):

        super(TestImageDataset, self).__init__()
        self.all_frames =  glob.glob(os.path.join(base_folder,'*.TIF'))
        self.all_frames.sort()


    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        idx = 39
        imagepath = self.all_frames[idx]
        imagename = imagepath.split('/')[-1].split('.')[0]
        print (imagepath) 
        img1 = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)

        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)) * 255
        img1 = np.array(img1, np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
        img1 = clahe.apply(img1)
        
        image1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))

        image1 = image1/torch.max(image1)

        #dataset return data
        output = {'image1': image1, 'name': imagename}

        return output
    
class TestFlowDataset(Dataset):

    def __init__(self,  base_folder = None ):

        super(TestFlowDataset, self).__init__()
        self.all_frames =  glob.glob(os.path.join(base_folder,'*.TIF'))
        self.all_frames.sort()


    def __len__(self):
        return len(self.all_frames ) -1

    def __getitem__(self, idx):

        imagepath1 = self.all_frames[len(self.all_frames ) -2 - idx]
        imagepath2 = self.all_frames[len(self.all_frames ) -1 - idx]
        #imagepath1 = self.all_frames[idx]
        #imagepath2 = self.all_frames[idx+1]
        print (imagepath1, imagepath2)
        img1 = cv2.imread(imagepath1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(imagepath2, cv2.IMREAD_UNCHANGED)

        #new changes for brightness change
        #img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)) * 255
        #img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * 255
        #img1 = (img1 / np.max(img1))* 255
        #img1 [img1 > 5] += 20
        #img1 = np.clip(img1,0,255)
        #img2 = (img2 / np.max(img2))* 255
        #img2 [img2 > 5] += 20
        #img2 = np.clip(img2,0,255)
        img1 = ( (img1 - 2) / 650) * 255
        img2 = ( (img2 - 2) / 650) * 255
        img1 = np.clip(img1,0,255)
        img2 = np.clip(img2,0,255)

        #increase contrast
        #print (np.unique(img1))
        img1 = np.array(img1, np.uint8)
        img2 = np.array(img2, np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
        img1 = clahe.apply(img1)
        img2 = clahe.apply(img2)
       
        image1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))
        image2 = torch.tensor(np.array(img2[None,:], dtype= np.float32))
        inp_image = torch.cat((image1,image2),  0)
            
        #normalizing the data to range 0 to 1
        inp_image = inp_image/torch.max(inp_image)
        image1 = image1/torch.max(image1)
        image2 = image2/torch.max(image2)

        #dataset return data
        output = {'images': inp_image, 'image1' : image1, 'image2': image2}

        return output


class CropImageDataset(Dataset):

    def __init__(self, cell_data, img, width = 96, height = 96):
        super(CropImageDataset).__init__()

        self.image = img
        self.cell_data = cell_data
        self.cell_ids = []

        self.h = height
        self.w = width

        H,W = self.image.shape

        #check for border cells
        for cell in cell_data.keys():
            (x,y) = self.cell_data[cell]['centroid']
            if x-int(self.h/2) >= 0 and y - int(self.w/2) >= 0 and x + int(self.h/2) < H and y + int(self.w/2) < W:
                self.cell_ids.append(cell)

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):

        cell = self.cell_ids[idx]

        (x,y) = self.cell_data[cell]['centroid']
        radius = self.cell_data[cell]['radius']
        #bounding box in required shape
        x1, y1 = x-int(self.h/2), y-int(self.w/2)
        x2, y2 = x+int(self.h/2), y+int(self.w/2)

        image = self.image[ y1:y2, x1:x2]
        image = cv2.resize(image,(224,224))
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image = np.moveaxis(image, -1, 0)

        image = image/ max(np.unique(image))
        
        image = torch.tensor(image, dtype= torch.float32)

        output = {'id': cell, 'image': image, 'bb': [x1,y1,x2,y2], 'centroid': [x,y], 'radius' : radius}

        return output





class TestMultiFlowDataset(Dataset):

    def __init__(self,  base_folder = None, max_interframe = 0, file_extension = 'tif' ):

        super(TestMultiFlowDataset, self).__init__()
        self.all_frames =  glob.glob(os.path.join(base_folder,'*.'+file_extension))
        
        self.all_frames.sort()

        self.allsets = {}
        self.create_all_pairs(max_interframe) 

        self.max_interframe = max_interframe


    def create_all_pairs(self, max_interframe = 0):
        '''
        create all the set of possible pairs within given maximum interframe distance
        '''
        for idx in range ( len(self.all_frames) -1):
            self.allsets[idx] = []
            for i in range(max_interframe+1):
                if idx+i+1 < len(self.all_frames):
                    self.allsets[idx].append([self.all_frames[idx], self.all_frames[idx+i+1]])

    def __len__(self):
        return len(self.allsets.keys())

    def getframelen(self):
        return len(self.all_frames) - 1

    def __getitem__(self, idx):

        curr_idx = len(self.allsets) - idx - 1

        imagepairs = []

        for i in range(self.max_interframe + 1):
            if i + curr_idx < len(self.all_frames) - 1 :
                imagepath1 = self.allsets[curr_idx][i][0]
                imagepath2 = self.allsets[curr_idx][i][1]

                print (imagepath1, imagepath2)
                
                filename1 = imagepath1.split('/')[-1].split('.')[0]
                filename2 = imagepath2.split('/')[-1].split('.')[0]
                
                img1 = cv2.imread(imagepath1, cv2.IMREAD_UNCHANGED)
                img2 = cv2.imread(imagepath2, cv2.IMREAD_UNCHANGED)
                

                #print (filename1, np.min(img1), np.max(img1) )
                #print (filename2, np.min(img2), np.max(img2) )

                #new changes for brightness change
                #img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)) * 255
                #img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * 255

                img1 = ( (img1 - 2) / 650) * 255
                img2 = ( (img2 - 2) / 650) * 255
                img1 = np.clip(img1,0,255)
                img2 = np.clip(img2,0,255)
                
            
                #increase contrast
                img1 = np.array(img1, np.uint8)
                img2 = np.array(img2, np.uint8)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
                img1 = clahe.apply(img1)
                img2 = clahe.apply(img2)

                image1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))
                image2 = torch.tensor(np.array(img2[None,:], dtype= np.float32))
                inp_image = torch.cat((image1,image2),  0)
                
                #normalizing the data to range 0 to 1
                inp_image = inp_image/torch.max(inp_image)
                image1 = image1/torch.max(image1)
                image2 = image2/torch.max(image2)

                #dataset return data
                output = {'interframe': i,'images': inp_image, 'image1' : image1, 'image2': image2, 'img1_name': filename1, 'img2_name': filename2}

                imagepairs.append(output)

        
        return imagepairs

