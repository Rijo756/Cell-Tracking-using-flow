import cv2, os
import numpy as np


class Preprocess():
    """
    This module takes in the input path and the output path and pre process all the images from the input path and stored to a folder in output path.
    This preprocessing step increases the brightness of cells so that cells can be easily detected.

    arguments:
        input_path : the folder path in which all the images are stored
        output_path : the results folder where the preprocessed images are stored as .png files into a folder "images"
        loghandler:   A loghandler object which can log the information into the logging file
    """

    def __init__(self, input_path = "" , output_path = "", loghandler = None):
        
        self.inputpath = input_path
        self.loghandler = loghandler
        self.outputpath = os.path.join(output_path,"images")
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

    def save_processed_images(self):
        '''
        read the images, apply the preprocessing and save the images
        '''
        files = os.listdir(self.inputpath)
        self.loghandler.log("info", str(len(files)) + " images found")
        for file in files:
            imagepath = os.path.join(self.inputpath,file)
            imagename = file.split(".")[0]

            image = cv2.imread(imagepath,cv2.IMREAD_UNCHANGED)

            #unint16 images process with saved values
            if image.dtype == np.uint16:
                image = ( (image - 2) / 650) * 255        
                image = np.clip(image,0,255)
            else:
                image = ((image - np.min(image))/( np.max(image) -  np.min(image)))*255

            cv2.imwrite(os.path.join(self.outputpath, imagename + ".png"), image)
                

def build_model(args, loghandler, resultfolder):
    '''
    This function sets all the values and calls the objects accordingly.
    '''
    inputpath = args.image_folder
    outputpath = os.path.join(args.output_folder, resultfolder)
    loghandler.log("info", "Starting the image preprocessing...")
    p = Preprocess(input_path=inputpath, output_path= outputpath, loghandler = loghandler)
    p.save_processed_images()
    loghandler.log("info", "Image preprocessing completed. Saved the images to: " + str(p.outputpath))