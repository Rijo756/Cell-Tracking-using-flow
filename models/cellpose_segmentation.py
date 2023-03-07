"""
For running cellpose it is written as in the same way as in the LivecellMiner by giving input image folder and output folder and sending a system command.
Since I have my cellpose module in a different conda environment.. This code activates and deactivates the environment during run. 
"""

import subprocess, os, glob, cv2, pickle
import numpy as np


def cellpose_segmentation(input_path, output_path):
    '''
    function to call the cellpose enviroment and get segmentation
    '''
    #change the cellpose environment path
    cmd = "/work/scratch/roy/miniconda3/envs/cellpose/bin/python -m cellpose --dir " + input_path + " --chan 0 --pretrained_model nuclei --savedir "+ output_path + " --diameter 30 --use_gpu --save_png"
    print ("Starting Cellpose Segmentaion...")
    subprocess.Popen(cmd, shell=True).wait()

    print ('Cellpose Instance Segmentation done and saved images')


def cellpose_data_creation(output_path, is_eliminate_boundarycells = False, is_eliminate_smallcells = True, is_eliminate_largecells = True, is_eliminate_lessdensecells = False):
    '''
    function to read the segmented images using cellpose and find the data for each cell 
    '''
    all_frames =  glob.glob(os.path.join(output_path,'*.png'))

    for frame in all_frames:
        img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        filename = str(frame).split('/')[-1].split('.')[0]
         #Get their cell inforamtions from watershed results for each cell
        cell_data = {}
        labels = np.array(img, np.uint8)
        unq = np.delete(np.unique(labels),0)
        for cell in unq:
            region = np.where(labels == cell)
            cellregion=np.zeros(labels.shape)
            cellregion[labels == cell]= 1

            #calcualte cell density of each cell to eliminate cells
            cell_density = int(np.sum(np.multiply(img, cellregion)))
            
            #bounding box, radius of cell inside bounding box and centroid of each cell
            bb = np.array([np.min(region[1]), np.min(region[0]), np.max(region[1]), np.max(region[0])]) 
            radius = int(np.max([bb[2] - bb[0], bb[3] - bb[1]]) / 2)
            centroid = (int(region[1].mean()), int(region[0].mean()))
            
            #to eliminate smaller cells
            if radius <= 8 and is_eliminate_smallcells:
                labels[labels==cell] = 0
                continue
            
            #check condition for borders and eliminate
            if is_eliminate_boundarycells:
                H,W = img.shape
                if centroid[0]-radius < -1 or centroid[0]+radius > W or centroid[1]-radius < -1 or centroid[1]+radius > H:
                    labels[labels==cell] = 0
                    continue

            #eliminate cells with less cell density
            if cell_density < 60 and is_eliminate_lessdensecells:
                labels[labels==cell] = 0
                continue

            #to eliminate bigger cells
            if radius > 45 and is_eliminate_largecells:
                labels[labels==cell] = 0
                continue
            
            #cell details with all the informations
            cell_data[cell] = {'bounding_box': bb, 'centroid': centroid, 'radius': radius, 'cell_density': cell_density}

        cv2.imwrite(frame, labels)
        with open(os.path.join(output_path, filename.split('.')[0] + '.pkl'), 'wb') as f:
            pickle.dump(cell_data, f)



def build_model(args, loghandler, resultfolder):
    '''
    This function sets all the values and calls the objects accordingly.
    '''
    inputpath = os.path.join(resultfolder,'images')
    outputpath = os.path.join(resultfolder,args.inst_segmentation)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    loghandler.log("info", "Starting the image Cellpose segmentaion...")
    cellpose_segmentation(input_path= inputpath, output_path=outputpath)
    loghandler.log("info", "Cellpose segmentation completed. Saved the images to: " + str(outputpath))
    loghandler.log("info", "Extracting cell data from cellpose segmentation...")
    cellpose_data_creation(output_path= outputpath)
    loghandler.log("info", "Extracted cell data from cellpose segmentation")
    