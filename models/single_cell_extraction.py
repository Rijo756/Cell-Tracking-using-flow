import os, glob, pickle, cv2, csv
import numpy as np


class Extract_Single_Cell():
    """
    A class to extract the cells which undergo cell splitting with the cell in center and save it to a different directory.

    arguments:
        loghandler      : A loghandler object which can log the information into the logging file
        trackpath       : The folder where the tracking results are stored
        imagepath       : The folder path where the images after preprocessing are stored
        outputpath      : The output path to which the results are stored
        before_split    : Number of frames needed to be considered before split
        after_split     : Number of frames needed to be considered after split
        resolution      : The resolution of the cropped image around each cell
        file_extension  : The file type or extension of the cell data stored files (pkl)

    """

    def __init__(self, loghandler, trackpath, imagepath, outputpath, before_split, after_split, resolution= (96,96), file_extension='pkl'):
        
        self.image_path = imagepath
        self.output_path = outputpath
        self.resolution = resolution
        self.loghandler = loghandler
        
        filepath = os.path.join(trackpath, 'aut_track.txt')
        self.track_details = self.read_track_details(filepath)

        self.before = before_split
        self.after = after_split
        self.find_possible_cases()
        #laod all pickled files
        self.store_data =  glob.glob(os.path.join(trackpath,'*.'+file_extension))
        self.store_data.sort()     
        #load all images
        self.all_frames =  glob.glob(os.path.join(imagepath,'*.png'))
        self.all_frames.sort()
        #load all tracked files
        self.all_masks =  glob.glob(os.path.join(trackpath,'*.png'))
        self.all_masks.sort()
    
    
    def read_track_details(self, filepath):
        '''
        function to read the tracking details from the file aut_track.txt
        '''
        track_details = {}

        with open(filepath, "r") as file:
            # loop through each line in the file
            for line in file:
                line = line.strip()
                values = [int(x) for x in line.split(",")]

                key = values[0]
                values = values[1:]
                track_details[key] = values
        return track_details


    def find_possible_cases(self):
        '''
        function to find the possible cases of cell splitting with the given number of frames before and after splitting
        '''
        self.crop_paths = []
        for cell in self.track_details.keys():
            if self.track_details[cell][2] != 0:
                parent_cell = self.track_details[cell][2]
                if self.track_details[parent_cell][1] - self.track_details[parent_cell][0] > self.before:
                    if self.track_details[cell][1] - self.track_details[cell][0] > self.after:
                        start = self.track_details[parent_cell][1]- self.before
                        end = self.track_details[cell][0] + self.after
                        self.crop_paths.append( [ parent_cell, start, self.track_details[parent_cell][1], cell,self.track_details[cell][0], end] ) 
        self.loghandler.log("info", "Found " + str(len(self.crop_paths))+ " cell tracks which undergo mitosis with given conditions."   )


    def extract_save_cell_paths(self):
        '''
        function to extract and save the cropped files in a folder structure
        '''
        crop_height = self.resolution[0]
        crop_width =  self.resolution[1]

        for x,crop_path in enumerate(self.crop_paths):

            cell_crop_details = []           
            #create folders for saving results
            outputpath1 = os.path.join(self.output_path, 'cell_' + "{0:03}".format(x+1) )
            self.loghandler.log("info", "Saving cell_" + "{0:03}".format(x+1)  )
            if not os.path.exists(outputpath1):
                os.makedirs(outputpath1)
            segmentationpath = os.path.join(outputpath1,'Mask')
            if not os.path.exists(segmentationpath):
                os.makedirs(segmentationpath)
            outputpath = os.path.join(outputpath1,'Raw')
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
            #to track the frame numbers
            frame_id = 1 
            #for loop for all the frames before 
            cell_id = crop_path[0]
            for i in range(crop_path[1],crop_path[2]+1):
                pkl_path = self.store_data[i-1]
                with open(pkl_path, 'rb') as f:
                    cell_data = pickle.load(f)

                (x,y) = cell_data[cell_id]['centroid']
                x1, y1 = x-int(crop_height/2), y-int(crop_width/2)
                x2, y2 = x+int(crop_height/2), y+int(crop_width/2)
                #load image , crop and save 
                imagepath = self.all_frames[i-1]
                image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
                img1 = image[ y1:y2, x1:x2]
                cell_crop_details.append([frame_id ,cell_id, x1.item(), y1.item(), x2.item(), y2.item(), imagepath] )
                image_name = os.path.join(outputpath, 'cell_id_'+ str(crop_path[3]) + '_track_id_'+ "{0:03}".format(frame_id)+'.png' )
                cv2.imwrite(image_name, img1) 
                #load segmentation mask crop and save
                maskpath = self.all_masks[i-1]
                mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
                msk1 = mask[ y1:y2, x1:x2]
                msk1[msk1 != cell_id] = 0
                msk1[msk1 == cell_id] = 255
                mask_name = os.path.join(segmentationpath, 'cell_id_'+ str(crop_path[3]) + '_track_id_' +"{0:03}".format(frame_id)+'.png' )
                cv2.imwrite(mask_name, msk1) 
                frame_id += 1
            #for loop for all the frames after 
            cell_id = crop_path[3]
            for i in range(crop_path[4],crop_path[5]+1):
                pkl_path = self.store_data[i-1]
                with open(pkl_path, 'rb') as f:
                    cell_data = pickle.load(f)

                (x,y) = cell_data[cell_id]['centroid']
                x1, y1 = x-int(crop_height/2), y-int(crop_width/2)
                x2, y2 = x+int(crop_height/2), y+int(crop_width/2)
                #load image and crop and save
                imagepath = self.all_frames[i-1]
                image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
                img1 = image[ y1:y2, x1:x2]
                image_name = os.path.join(outputpath, 'cell_id_'+ str(crop_path[3]) + '_track_id_'+ "{0:03}".format(frame_id)+'.png' )
                cv2.imwrite(image_name, img1) 
                #load segmentation mask crop and save
                maskpath = self.all_masks[i-1]
                mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
                msk1 = mask[ y1:y2, x1:x2]
                cell_crop_details.append([frame_id ,cell_id, x1.item(),y1.item(), x2.item(), y2.item(), imagepath] )
                msk1[msk1 != cell_id] = 0
                msk1[msk1 == cell_id] = 255
                mask_name = os.path.join(segmentationpath, 'cell_id_'+ str(crop_path[3]) + '_track_id_'+ "{0:03}".format(frame_id)+'.png' )
                cv2.imwrite(mask_name, msk1) 
                frame_id += 1
            #write the cropped details into a csv file
            file_path = os.path.join(outputpath1, 'crop_details.csv')
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                for row in cell_crop_details:
                    writer.writerow(row)



def build_model(args, loghandler, resultfolder, devicetype):
     
    image_path = os.path.join(resultfolder, 'images')
    suffix = '_postprocess' if args.is_tracking_postprocess else ''
    track_path = os.path.join(resultfolder, args.tracking_method + suffix)
    outputpath = os.path.join(resultfolder, 'single_cells')
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    before_split = args.number_frames_before_split
    after_split = args.number_frames_after_split
    
    loghandler.log("info", "Starting single cell extraction which undergoes mitosis and detected by tracking method...")

    e = Extract_Single_Cell(loghandler, track_path, image_path, outputpath, before_split, after_split)
    e.extract_save_cell_paths()
    loghandler.log("info", "Single cell extraction cmpleted. Saved results to folder single_cells")
