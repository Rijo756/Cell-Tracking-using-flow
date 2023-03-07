import os, glob, torch, cv2, pickle, math, itertools  
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import ImageColor
from random import randint
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from .unet_architecture import UNetArchitecture

class TrackDataset(Dataset):
    '''
    The dataset creates pairs of images and use it for predicting. 

    The pairs are created in list based on max_interframe value. max_interframe decides until how many nearest neighbour pairs should be created.

    arguments: 
        base_folder:            The path to the folder which contains all the images
        segmentation_folder:    The path to the folder which contains all the segmentation mask to images
        max_interframe:         The number of neighbor frames needed to be considered for tracking 0 means only nearest frames and value of 1 means two nearest frames. 
                                     Higher values greater than 2 might bring error results if cell movements are large.
        file_extension:         The format of the images and the segmentations stored
    '''
    def __init__(self, base_folder = None, segmentation_folder = None, max_interframe = 0, file_extension = 'png'):
        self.all_frames =  glob.glob(os.path.join(base_folder,'*.'+file_extension))        
        self.all_frames.sort()
        self.inputpath = base_folder
        self.segmentation_folder = segmentation_folder
        self.allsets = {}
        self.create_all_pairs(max_interframe) 
        self.max_interframe = max_interframe
        
    def create_all_pairs(self, max_interframe = 0):
        '''
        create all the set of possible pairs of images within given maximum interframe distance
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
        #create a list of values based on maximum interframe distance
        for i in range(self.max_interframe + 1):
            if i + curr_idx < len(self.all_frames) - 1 :
                imagepath1 = self.allsets[curr_idx][i][0]
                imagepath2 = self.allsets[curr_idx][i][1]                
                filename1 = imagepath1.split('/')[-1].split('.')[0]
                filename2 = imagepath2.split('/')[-1].split('.')[0]
                img1 = cv2.imread(imagepath1, cv2.IMREAD_UNCHANGED)
                img2 = cv2.imread(imagepath2, cv2.IMREAD_UNCHANGED)

                labels1 = cv2.imread(os.path.join(self.segmentation_folder,filename1 + '.png'), cv2.IMREAD_UNCHANGED)
                labels2 = cv2.imread(os.path.join(self.segmentation_folder,filename2 + '.png'), cv2.IMREAD_UNCHANGED)

                with open(os.path.join(self.segmentation_folder,filename1 + '.pkl'), 'rb') as f:
                    cell1_data = pickle.load(f)
                with open(os.path.join(self.segmentation_folder,filename2 + '.pkl'), 'rb') as f:
                    cell2_data = pickle.load(f)

                image1 = torch.tensor(np.array(img1[None,:], dtype= np.float32))
                image2 = torch.tensor(np.array(img2[None,:], dtype= np.float32))
                inp_image = torch.cat((image1,image2),  0)
                
                #normalizing the data to range 0 to 1
                inp_image = inp_image/torch.max(inp_image)
                image1 = image1/torch.max(image1)
                image2 = image2/torch.max(image2)

                #dataset return data
                output = {'interframe': i,'images': inp_image, 'image1' : image1, 'cell1_data': cell1_data, 'labels1': labels1, 'img1_name': filename1, 
                                 'image2': image2, 'cell2_data': cell2_data, 'labels2': labels2,'img2_name': filename2}
                                 
                imagepairs.append(output)
        return imagepairs




class Tracking():
    '''
    Parent class for tracking, contains some common fuctions and variables for tracking

    arguments:
        loghandler:     A loghandler object which can log the information into the logging file
        input_path:     The folder path where the images are stored
        seg_path:       The folder path where the segmentation masks are stored for the images
        file_type:      The format of the images being stored
        output_path:    The folder path where the tracking results are stored
        visualize_path: The folder path to visualize the results
        max_interframe: Number of nearest frames needed to be considered for tracking
                         0 means only nearest frames and value of 1 means two nearest frames. 
                         Higher values greater than 2 might bring error results if cell movements are large.
        batch_size:     The batch size for the dataloader. Better keep the batch size 1.
    '''
    def __init__(self, loghandler, input_path, seg_path, file_type, output_path, visualize_path = None, max_interframe = 0, batch_size = 1):
        self.loghandler = loghandler
        self.inputpath = input_path
        self.seg_path = seg_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.file_extension = file_type
        self.max_interframe = max_interframe #max_interframe 0 means only nearby frames, 1 means upto 2 nearest
        self.dataset = TrackDataset(base_folder = self.inputpath, segmentation_folder= self.seg_path ,max_interframe = self.max_interframe, file_extension = file_type)
        self.dataloader = DataLoader(self.dataset, batch_size= self.batch_size, shuffle= False)
        self.new_cellid = -1
        self.track_cell_details = {'New':{}, 'Lost':{}, 'Mitosis':{}, 'Parent':{}}
        self.cell_data = {}
        self.visualization_path = visualize_path
        self.color = []
        for i  in range(300):
            self.color.append(ImageColor.getcolor( '#%06X' % randint(100, 0xFFFFFF), "RGB") )

    def save_tracking_results(self, orig_image, labels, cell_data, filename):
        '''
        function to save the track results both the image and the cell data
        '''
        labels = np.array(labels,dtype=np.uint8)
        cv2.imwrite(os.path.join(self.output_path, filename+'.png'), labels)
        with open(os.path.join(self.output_path, filename.split('.')[0] + '.pkl'), 'wb') as f:
            pickle.dump(cell_data, f)
        self.visualize_frames(orig_image, labels, filename, cell_data)

    def visualize_frames(self, orig_image, labels, filename, cell_data):
        '''
        function to visualize the track results with colors and cell numbers(for better understanding)
        '''
        #visualize instance sementation
        trackimage = np.zeros((labels.shape[0], labels.shape[1], 3))
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB) * 255
        unq = np.delete(np.unique(labels),0)
        for i,u in enumerate(unq):
            try:
                trackimage[:,:,0][labels == u] = self.color[u][0]
                trackimage[:,:,1][labels == u] = self.color[u][1]
                trackimage[:,:,2][labels == u] = self.color[u][2]
            except:
                print ('Error: Cell number ',u, ' greater than declared number of colors. Increase colors')
                print (1/0)

            if u in cell_data.keys():
                x1, y1 = cell_data[u]['centroid']
                x1 = x1.item()
                y1 = y1.item()
                rgb_image = cv2.putText(rgb_image, str(u), [x1,y1], cv2.FONT_HERSHEY_SIMPLEX,  1.0, (255, 255, 255), 2, cv2.LINE_AA)

        trackimage = np.concatenate((rgb_image,trackimage),axis = 1)
        cv2.imwrite(os.path.join(self.visualization_path, filename+'.png'), trackimage)

    def update_celldata2_list(self, cell_data2_list, new_cell_data1):
        '''
        function to update tracked results for the nearest n frames (max_interframe)
        '''
        if len(cell_data2_list) < (self.max_interframe + 1):
            cell_data2_list.insert(0,new_cell_data1)
        else:
            cell_data2_list.pop()
            cell_data2_list.insert(0,new_cell_data1)

    def get_track_data(self):
        '''
        function to find and save for each cell their start frame end frame and parent cell.
        '''
        aut_track = []
        track_sorted = {k: v for k, v in sorted(self.track_cell_details['New'].items(), key=lambda item: item[1])}
        for cell in track_sorted.keys():
            data  = [cell,self.track_cell_details['New'][cell],-1,0]
            if cell in self.track_cell_details['Lost'].keys():
                data[2] = self.track_cell_details['Lost'][cell] - 1
            else:
                data[2] = len(self.dataloader) + 1
                self.track_cell_details['Lost'][cell] = data[2] + 1
            #parent cell
            if cell in self.track_cell_details['Parent'].keys():
                data[3] = self.track_cell_details['Parent'][cell]
            aut_track.append(data)
        with open(os.path.join(self.output_path,'aut_track.txt'), 'w') as f:
            for line in aut_track:
                f.write(f"{str(line[0])+','+str(line[1])+','+str(line[2])+','+str(line[3])}\n")

    def update_tracking_results_postprocess(self, resultspath, visualizepath):
        '''
        function to update the tracking results by removing boundary cells, large cells and discontinues cells.
        '''
        trackpath = self.output_path
        all_frames =  glob.glob(os.path.join(trackpath,'*.'+self.file_extension))        
        all_frames.sort()

        #remove boundary cells, remove large cells
        removed_cells_frames = {}
        img = cv2.imread(all_frames[0], cv2.IMREAD_UNCHANGED)
        H,W = img.shape
        for i in range(len(self.dataloader)+1):
            removed_cells_frames[i] = []
            cell_keys = list(self.cell_data[i+1].keys())
            for cell in cell_keys:
                if cell in self.cell_data[i+1].keys():
                    x1,y1 = self.cell_data[i+1][cell]['centroid']
                    x1 = x1.item()
                    y1 = y1.item()
                    radius = self.cell_data[i+1][cell]['radius']
                    radius = radius.item()
                    if x1-radius < -1 or x1+radius > W or y1-radius < -1 or y1+radius > H: #boundary condition
                        del self.cell_data[i+1][cell]
                        removed_cells_frames[i].append(cell)
                        if cell in self.track_cell_details['New'].keys():
                            del self.track_cell_details['New'][cell]
                    elif radius > 45: #large cell condition
                        del self.cell_data[i+1][cell]
                        removed_cells_frames[i].append(cell)
                        if cell in self.track_cell_details['New'].keys():
                            del self.track_cell_details['New'][cell]

        #remove the discontinues cells
        track_sorted = {k: v for k, v in sorted(self.track_cell_details['New'].items(), key=lambda item: item[1])}
        cell_ids = track_sorted.keys()
        fulltime_cells = []
        aut_track = []
        for cell in cell_ids:
            start = track_cellstart(cell, self.track_cell_details)
            lost =  track_celllost(cell, self.track_cell_details)
            if start==1 and lost == len(self.dataloader)+1:
                data  = [cell, self.track_cell_details['New'][cell],-1,0]
                if cell in self.track_cell_details['Lost'].keys():
                    data[2] = self.track_cell_details['Lost'][cell] - 1
                else:
                    data[2] = len(self.dataloader)+1
                #parent cell
                if cell in self.track_cell_details['Parent'].keys():
                    data[3] = self.track_cell_details['Parent'][cell]
                aut_track.append(data) 
                fulltime_cells.append(cell)

        #store and visualize the postprocess results
        self.output_path = resultspath
        self.visualization_path = visualizepath
        for i,files in enumerate(all_frames):
            trackimage = cv2.imread(files, cv2.IMREAD_UNCHANGED)
            filename = files.split('/')[-1]
            cell_data = self.cell_data[i+1]
            unq = np.delete(np.unique(trackimage),0)
            for u in unq:
                if u not in fulltime_cells:
                    trackimage[trackimage==u] = 0
                    if u in cell_data.keys():
                        del cell_data[u]
            orig_image = cv2.imread(os.path.join(self.inputpath, filename), cv2.IMREAD_UNCHANGED)
            orig_image = np.array(orig_image/255, dtype= np.float32)
            filename =  filename.split(".")[0]
            self.save_tracking_results(orig_image, trackimage, cell_data, filename)

        #save the postprocessed cell info into aut_track.txt
        aut_track = []
        for cell in fulltime_cells:
            data  = [cell,self.track_cell_details['New'][cell],-1,0]
            if cell in self.track_cell_details['Lost'].keys():
                data[2] = self.track_cell_details['Lost'][cell] - 1
            else:
                data[2] = len(self.dataloader) + 1
                self.track_cell_details['Lost'][cell] = data[2] + 1
            #parent cell
            if cell in self.track_cell_details['Parent'].keys():
                data[3] = self.track_cell_details['Parent'][cell]
            aut_track.append(data)
        with open(os.path.join(self.output_path,'aut_track.txt'), 'w') as f:
            for line in aut_track:
                f.write(f"{str(line[0])+','+str(line[1])+','+str(line[2])+','+str(line[3])}\n")


def track_cellstart(cell, track_cell_details):
    '''
    recursive function till start to find cell lineage
    '''
    if cell in track_cell_details['Parent'].keys():
        parent_cell = track_cell_details['Parent'][cell]
        if parent_cell in track_cell_details['New'].keys():
            return track_cellstart(parent_cell, track_cell_details)
        else:
            return -1
    else:
        return track_cell_details['New'][cell]

def track_celllost(cell, track_cell_details):
    '''
    recursive function till end to find cell lineage
    '''
    if cell in track_cell_details['Mitosis'].keys():
        daughter1 = track_cell_details['Mitosis'][cell][0]
        daughter2 = track_cell_details['Mitosis'][cell][1]
        return max(track_celllost(daughter1, track_cell_details), track_celllost(daughter2, track_cell_details))
    else:
        if cell in track_cell_details['Lost'].keys():
            return track_cell_details['Lost'][cell] - 1
        else:
            print ('Error: ',cell, ' not lost.')
            assert(cell in track_cell_details['Lost'].keys())


class NearestNeighbor(Tracking):
    '''
    Tracking using Nearest neighbor approach between nearby cells

    arguments:
        loghandler:     A loghandler object which can log the information into the logging file
        input_path:     The folder path where the images are stored
        seg_path:       The folder path where the segmentation masks are stored for the images
        file_type:      The format of the images being stored
        output_path:    The folder path where the tracking results are stored
        visualize_path: The folder path to visualize the results
        max_interframe: Number of nearest frames needed to be considered for tracking
                         0 means only nearest frames and value of 1 means two nearest frames. 
                         Higher values greater than 2 might bring error results if cell movements are large.
        batch_size:     The batch size for the dataloader. Better keep the batch size 1.
    '''

    def __init__(self, loghandler, input_path, seg_path, file_type, output_path, visualize_path, max_interframe, batch_size):
        super().__init__(loghandler, input_path = input_path, seg_path= seg_path, file_type = file_type, output_path = output_path, visualize_path = visualize_path ,max_interframe = max_interframe, batch_size = batch_size)
        

    def nearest_neighbour_track(self):
        '''
        main function which goes through the images from dataloader and apply nearest neighbour for each frame
        '''
        cell_data_current = None
        cell_data2_list = []
        with torch.no_grad():
            #=====Iteration over mini batches=====
            for didx,data in enumerate(self.dataloader):

                frame_id = len(self.dataloader)- (didx* self.batch_size)   
                
                if didx == 0:
                    #the first pair of image has only one nearest neighbour because it is the first cell on which we evaluate.
                    labels2 = data[0]['labels2'][0].detach().numpy()
                    orig_image = data[0]['image2'][0][0].detach().numpy()
                    cell_data_current = data[0]['cell2_data']
                    filename2 = data[0]['img2_name'][0]
                    self.cell_data[frame_id+1] = cell_data_current
                    self.save_tracking_results(orig_image,labels2, cell_data_current, filename2)
                    cell_data2_list = [ data[x]['cell2_data'] for x in range(len(data)) ]

                labels1 = data[0]['labels1'][0].detach().numpy()
                orig_image = data[0]['image1'][0][0].detach().numpy()
                cell_data1 =  data[0]['cell1_data']
                filename1 = data[0]['img1_name'][0]

                new_labels1, new_cell_data1 = self.nearest_neighbour_approach(labels1, cell_data1, cell_data2_list, cell_data_current, frame_id)
                cell_data_current = new_cell_data1
                self.cell_data[frame_id] = cell_data_current
                self.save_tracking_results(orig_image, new_labels1, cell_data_current, filename1)
                self.update_celldata2_list(cell_data2_list, new_cell_data1)
                #print (filename1)
                self.loghandler.log('info', "\t Tracking image: " + str(filename1))
        #all cells in the first are new
        for cell in cell_data_current.keys():
            self.track_cell_details['New'][cell] = frame_id
        self.get_track_data()

    def nearest_neighbour_approach(self, labels1, cell_data1, cell_data2_list, cell_data_current, frame_id):
        '''
        function to find the nearest neighbour of each cell in given frame
        '''
        new_labels1 = np.zeros(labels1.shape, dtype=np.int32)
        new_cell_data1 = {}
        covered_ids1 = []

        cell2_keys = cell_data_current.keys()
        #check for each cell from last(current) frame with cells in this frame
        for cell2 in cell2_keys:
        
            temp_data = None
            temp_cellid = -1
            min_d  = 38
            possible_cell1 = []
            for cell1 in cell_data1.keys():
                x1,y1 = cell_data1[cell1]["centroid"]
                x1 = x1.item()
                y1 = y1.item()
                for i in range(self.max_interframe+1):
                    if (frame_id + i) <= len(self.dataloader):
                        if cell2 not in cell_data2_list[i].keys():
                            continue
                        x2,y2 = cell_data2_list[i][cell2]["centroid"]
                        x2 = x2.item()
                        y2 = y2.item()
                        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if d< min_d:
                            possible_cell1.append([d,cell1])
            
            if len(possible_cell1)>0:
                possible_cell1 = sorted(possible_cell1, key=lambda x: x[0])
                temp_cellid = possible_cell1[0][1] #max(set(possible_cell1), key = possible_cell1.count)
                covered_ids1.append(temp_cellid)
                temp_data = cell_data1[temp_cellid]
                new_cell_data1[cell2] = temp_data
                new_labels1[labels1==temp_cellid] = int(cell2)

        #cells joining
        for cell1 in cell_data1.keys():
            if cell1 not in covered_ids1:
                temp_data = cell_data1[cell1]
                if self.new_cellid == -1:
                    new_id = max(new_cell_data1.keys()) + 1
                    self.new_cellid = new_id
                else:
                    self.new_cellid = self.new_cellid + 1
                    new_id = self.new_cellid
                new_cell_data1[new_id] = temp_data
                new_labels1[labels1==cell1] = int(new_id)
                self.track_cell_details['Lost'][new_id] = frame_id+1
        
        #detect parent cells and daughter cells
        cell_list = list(itertools.combinations(new_cell_data1.keys(), r=2))
        for c1, c2 in cell_list:
            if c1 in new_cell_data1.keys() and c2 in new_cell_data1.keys():
                if new_cell_data1[c1]['centroid'] == new_cell_data1[c2]['centroid'] and new_cell_data1[c1]['radius'] == new_cell_data1[c2]['radius']:
                    temp_data = new_cell_data1[c1]
                    if self.new_cellid == -1:
                        new_id = max(new_cell_data1.keys()) + 1
                        self.new_cellid = new_id
                    else:
                        self.new_cellid = self.new_cellid + 1
                        new_id = self.new_cellid
                    new_labels1[new_labels1==c2] = int(new_id)
                    new_labels1[new_labels1==c1] = int(new_id)
                    del new_cell_data1[c1]
                    del new_cell_data1[c2]
                    new_cell_data1[new_id] = temp_data            
                    self.track_cell_details['Lost'][new_id] = frame_id+1
                    self.track_cell_details['Mitosis'][new_id] = [c1,c2]
                    self.track_cell_details['Parent'][c1] = new_id
                    self.track_cell_details['Parent'][c2] = new_id
        
        #new cells
        for cell2 in cell2_keys:
            if cell2 not in new_cell_data1.keys():
                self.track_cell_details['New'][cell2] = frame_id+1
    
        return new_labels1, new_cell_data1



class UNetFlowTracking(Tracking):
    '''
    Tracking using UNet flow method 

    arguments:
        loghandler:             A loghandler object which can log the information into the logging file
        pretrained_model_path:  The file path to the saved model for the UNet flow
        input_path:             The folder path where the images are stored
        seg_path:               The folder path where the segmentation masks are stored for the images
        file_type:              The format of the images being stored
        output_path:            The folder path where the tracking results are stored
        visualize_path:         The folder path to visualize the results
        max_interframe:         Number of nearest frames needed to be considered for tracking
                                    0 means only nearest frames and value of 1 means two nearest frames. 
                                    Higher values greater than 2 might bring error results if cell movements are large.
        batch_size:             The batch size for the dataloader. Better keep the batch size 1.
        device_type:            To predict using either cpu or gpu
        input_channels:         The number of input channels in the flownetwork input (2 channels - one for each image)
        depth:                  Depth of the UNet Architecture (default: 3)
    '''

    def __init__(self, loghandler, pretrained_model_path, input_path, seg_path, file_type, output_path, visualize_path, max_interframe, batch_size, device_type, input_channels = 2, depth = 3):
        super().__init__(loghandler, input_path = input_path, seg_path= seg_path, file_type = file_type, output_path = output_path, visualize_path = visualize_path ,max_interframe = max_interframe, batch_size = batch_size)
        self.device_type = device_type
        self.model= UNetArchitecture(num_classes = 1, in_channels= input_channels, depth= depth)
        trained_model_s = torch.load(pretrained_model_path)
        self.model.load_state_dict(trained_model_s)
        self.model = self.model.to(torch.device(self.device_type))

    def pred_flow(self, input_data):
        '''
        For a given frame find the flow for all the nearest n frames from amx_interframes and save it to a list  
        '''
        flow_list = []
        for inp in input_data:
            inp = inp.to(device= self.device_type)
            flow = self.test_time_augmentations(inp)
            flow_post = self.watershed_algorithm(flow[0][0].cpu().detach().numpy())
            flow_list.append(flow_post)
        return flow_list

    def test_time_augmentations(self, tensor_input):
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

    def watershed_algorithm(self, pred_mask):
        '''
        Apply watershed algorithm over the flow results to seperate the cell flows.
        '''
        #NORMALIZE
        pred_mask = ((pred_mask - np.min(pred_mask))/ (np.max(pred_mask) - np.min(pred_mask)) )*255

        ##thresholding
        _, thresh = cv2.threshold(pred_mask, 10, 255, cv2.THRESH_BINARY)
        thresh = np.array(thresh,dtype=np.uint8)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        
        #distance map and local maximas    
        thresh = opening
        D = ndi.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=25,labels=thresh)

        #watershed
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        #dialate the segmentations
        unq = np.delete(np.unique(labels),0)
        for cell in unq:
            cellregion=np.zeros(labels.shape)
            cellregion[labels == cell]=1
            cellregion = cv2.morphologyEx(cellregion, cv2.MORPH_DILATE, kernel, iterations=2)
            labels[(labels== 0) & (cellregion==1)] = cell

        return labels

    def unetflow_tracking(self):
        '''
        main function which goes through the images from dataloader and apply flow based tracking for each frame
        '''
        cell_data_current = None
        cell_data2_list = []
        
        with torch.no_grad():
            #=====Iteration over mini batches=====
            for didx,data in enumerate(self.dataloader):
                self.model.eval()
                frame_id = len(self.dataloader)- (didx* self.batch_size)    

                if didx == 0:
                    #the first pair of image has only one nearest neighbour because it is the first cell on which we evaluate.
                    labels2 = data[0]['labels2'][0].detach().numpy()
                    orig_image = data[0]['image2'][0][0].detach().numpy()
                    cell_data_current = data[0]['cell2_data']
                    filename2 = data[0]['img2_name'][0]
                    self.cell_data[frame_id+1] = cell_data_current
                    self.save_tracking_results(orig_image,labels2, cell_data_current, filename2)
                    cell_data2_list = [ data[x]['cell2_data'] for x in range(len(data)) ]
                    
                #get flow
                inputdata = [ data[x]['images'] for x in range(len(data)) ]
                flow_list = self.pred_flow(inputdata)
    
                labels1 = data[0]['labels1'][0].detach().numpy()
                orig_image = data[0]['image1'][0][0].detach().numpy()
                cell_data1 =  data[0]['cell1_data']
                filename1 = data[0]['img1_name'][0]

                new_labels1, new_cell_data1 = self.flow_based_approach(labels1, cell_data1, cell_data2_list,flow_list, cell_data_current, frame_id)
                cell_data_current = new_cell_data1
                self.cell_data[frame_id] = cell_data_current
                self.save_tracking_results(orig_image, new_labels1, cell_data_current, filename1)
                self.update_celldata2_list(cell_data2_list, new_cell_data1)
                #print (filename1)
                self.loghandler.log('info', "\t Tracking image: " + str(filename1))
        #all cells in the first are new
        for cell in cell_data_current.keys():
            self.track_cell_details['New'][cell] = frame_id
        self.get_track_data()

    def flow_based_approach(self,labels1, cell_data1, cell_data2_list, flow_list, cell_data_current, frame_id):
        '''
        function to find the flow based nearest neighbor of each cell in given frame
        '''
        new_labels1 = np.zeros(labels1.shape, dtype=np.int32)
        new_cell_data1 = {}
        covered_ids1 = []

        cell2_keys = cell_data_current.keys()
        #check for each cell from last(current) frame with cells in this frame
        for cell2 in cell2_keys:
        
            temp_data = None
            temp_cellid = -1
            min_d  = 15
            possible_cell1 = []
            for cell1 in cell_data1.keys():
                x1,y1 = cell_data1[cell1]["centroid"]
                x1 = x1.item()
                y1 = y1.item()
                for i in range(self.max_interframe+1):
                    if (frame_id + i) <= len(self.dataloader):
                        if cell2 not in cell_data2_list[i].keys():
                            continue
                        x2,y2 = cell_data2_list[i][cell2]["centroid"]
                        x2 = x2.item()
                        y2 = y2.item()
                        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        flow_labels = flow_list[i]
                        if np.array_equal( flow_labels[y1,x1], flow_labels[y2,x2]) and not np.array_equal( flow_labels[y1,x1], np.array([0,0,0])) and flow_labels[y1,x1] != 0:
                            possible_cell1.append(cell1)
                        elif d< min_d:
                            possible_cell1.append(cell1)
            
            if len(possible_cell1)>0:
                temp_cellid = max(set(possible_cell1), key = possible_cell1.count)
                covered_ids1.append(temp_cellid)
                temp_data = cell_data1[temp_cellid]
                new_cell_data1[cell2] = temp_data
                new_labels1[labels1==temp_cellid] = int(cell2)
            else: #missing cell cases:
                x2,y2 = cell_data_current[cell2]["centroid"]
                x2 = x2.item()
                y2 = y2.item()
                dist = 10000
                temp_cell = -1
                for cell1 in cell_data1.keys():
                    x1,y1 = cell_data1[cell1]["centroid"]
                    x1 = x1.item()
                    y1 = y1.item()
                    if dist > math.sqrt((y2-y1)**2 + (x2-x1)**2):
                        dist = math.sqrt((y2-y1)**2 + (x2-x1)**2)
                        temp_cell = cell1
                if dist < 38 and temp_cell != -1:
                    covered_ids1.append(temp_cell)
                    temp_data = cell_data1[temp_cell]
                    new_cell_data1[cell2] = temp_data
                    new_labels1[labels1==temp_cell] = int(cell2)

        #cells joining
        for cell1 in cell_data1.keys():
            if cell1 not in covered_ids1:
                temp_data = cell_data1[cell1]
                if self.new_cellid == -1:
                    new_id = max(new_cell_data1.keys()) + 1
                    self.new_cellid = new_id
                else:
                    self.new_cellid = self.new_cellid + 1
                    new_id = self.new_cellid
                new_cell_data1[new_id] = temp_data
                new_labels1[labels1==cell1] = int(new_id)
                self.track_cell_details['Lost'][new_id] = frame_id+1
        
        #detect parent cells and daughter cells
        cell_list = list(itertools.combinations(new_cell_data1.keys(), r=2))
        for c1, c2 in cell_list:
            if c1 in new_cell_data1.keys() and c2 in new_cell_data1.keys():
                if new_cell_data1[c1]['centroid'] == new_cell_data1[c2]['centroid'] and new_cell_data1[c1]['radius'] == new_cell_data1[c2]['radius']:
                    temp_data = new_cell_data1[c1]
                    if self.new_cellid == -1:
                        new_id = max(new_cell_data1.keys()) + 1
                        self.new_cellid = new_id
                    else:
                        self.new_cellid = self.new_cellid + 1
                        new_id = self.new_cellid
                    new_labels1[new_labels1==c2] = int(new_id)
                    new_labels1[new_labels1==c1] = int(new_id)
                    del new_cell_data1[c1]
                    del new_cell_data1[c2]
                    new_cell_data1[new_id] = temp_data            
                    self.track_cell_details['Lost'][new_id] = frame_id+1
                    self.track_cell_details['Mitosis'][new_id] = [c1,c2]
                    self.track_cell_details['Parent'][c1] = new_id
                    self.track_cell_details['Parent'][c2] = new_id
        
        #new cells
        for cell2 in cell2_keys:
            if cell2 not in new_cell_data1.keys():
                self.track_cell_details['New'][cell2] = frame_id+1
    
        return new_labels1, new_cell_data1


def build_model(args, loghandler, resultfolder, devicetype):
    '''
    This function sets all the values and calls the objects accordingly.
    '''
    inputpath = os.path.join(resultfolder, 'images')
    segmentation_path = os.path.join(resultfolder, args.inst_segmentation)
    filetype = 'png'
    outputpath = os.path.join(resultfolder,args.tracking_method)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    visualpath = os.path.join(resultfolder,args.tracking_method + '_visualized')
    if not os.path.exists(visualpath):
        os.makedirs(visualpath)
    batchsize = 1  #batch size should be set to 1 for no errors.
    max_interframe = 0
    loghandler.log("info", "Starting " + str(args.tracking_method) +" based tracking of cells...")
    if args.tracking_method == 'nearestneighbor':
        trck = NearestNeighbor(loghandler, input_path = inputpath, seg_path= segmentation_path , file_type = filetype, output_path = outputpath, visualize_path = visualpath, max_interframe = max_interframe, batch_size= batchsize)
        trck.nearest_neighbour_track()
    elif args.tracking_method == 'unetflow':
        trck = UNetFlowTracking (loghandler, pretrained_model_path= args.pretrained_unetflow_model ,input_path = inputpath, seg_path= segmentation_path , file_type = filetype, output_path = outputpath, visualize_path = visualpath, max_interframe = max_interframe, batch_size= batchsize, device_type= devicetype)
        trck.unetflow_tracking()
    loghandler.log("info", "Completed " + str(args.tracking_method) +" based tracking of cells. Saved results to folder "+ args.tracking_method)
    loghandler.log("info", "Starting tracking post processing to remove some cells...")
    #tracking post process steps
    if args.is_tracking_postprocess:
        postprocess_resultpath = os.path.join(resultfolder,args.tracking_method + '_postprocess')
        if not os.path.exists(postprocess_resultpath):
            os.makedirs(postprocess_resultpath)
        postprocess_visualpath = os.path.join(resultfolder,args.tracking_method + '_postprocess_visualized')
        if not os.path.exists(postprocess_visualpath):
            os.makedirs(postprocess_visualpath)
        trck.update_tracking_results_postprocess(postprocess_resultpath, postprocess_visualpath)
        loghandler.log("info", "Completed tracking post processing steps. Saved results to folder "+ args.tracking_method + '_postprocess')