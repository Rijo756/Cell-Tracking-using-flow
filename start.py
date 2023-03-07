import datetime, os, torch
import argparse

from models import image_preprocessing, cellpose_segmentation, unet_seg_watershed, tracking_methods, imagenet_classification_methods, single_cell_extraction, single_cell_classification
from utils.LogHandler import LogHandler

#defining arguments for training
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--image_folder',type=str, default="/home/students/roy/trackingcells/images_test/01", help="Path to the input image folder")
parser.add_argument('--output_folder',type=str, default="/home/students/roy/Cell_Track_Classify/output", help="Path of the output folder, inside this folder the result folder is created")

#tracking
parser.add_argument('--inst_segmentation',type=str, default="unetwatershed", help="Method for instance segmentation select from: ['cellpose','unetwatershed']")
parser.add_argument('--tracking_method',type=str, default="unetflow", help="Method for tracking select from: ['nearestneighbor','unetflow']")
parser.add_argument('--is_tracking_postprocess',type=bool, default=True, help="Whether to do tracking postprocess to remove non continues cells")
parser.add_argument('--pretrained_unetsegmentaion_model',type=str, default="/home/students/roy/Cell_Track_Classify/pretrained_models/seg_network_trained_Hela_aug_50.pth", help="full path to the pretrained segmentation model")
parser.add_argument('--pretrained_unetflow_model',type=str, default="/home/students/roy/Cell_Track_Classify/pretrained_models/flow_network_trained_Hela_aug_100.pth", help="full path to the pretrained flow model")

#Detect and extract mitosis tracks and classify those tracks
parser.add_argument('--is_extract_mitosis_tracks',type=bool, default=True, help="Whether to extract single cells undergoing mitosis")
parser.add_argument('--number_frames_before_split',type=int, default=20, help="Number of cells before mitosis splitting")
parser.add_argument('--number_frames_after_split',type=int, default=20, help="Number of cells after mitosis splitting")
parser.add_argument('--is_classify_single_cell_tracks',type=bool, default=True, help="Whether to classifiy the extracted single cell tracks")
parser.add_argument('--single_cell_classification_method',type=str, default="basemodel", help="Method for tracking select from: ['resnet18','basemodel','timeencodedresnet']")
parser.add_argument('--pretrained_single_cell_clf_model',type=str, default="/home/students/roy/Cell_Track_Classify/pretrained_models/BaseModel_pretrained_model_LSM710.pth", help="full path to the pretrained classification model")

#classification of all cells
parser.add_argument('--is_classify_cells',type=bool, default=True, help="Whether to classify all the cells after tracking")
parser.add_argument('--classification_method',type=str, default="resnet18", help="Method for tracking select from: ['resnet18','basemodel','timeencodedresnet']")
parser.add_argument('--pretrained_clf_model',type=str, default="/home/students/roy/Cell_Track_Classify/pretrained_models/ResNet18_pretrained_model_LSM710.pth", help="full path to the pretrained classification model")

args = parser.parse_args()

if __name__ == "__main__":

    #select gpu or cpu
    device_type = "cuda" if torch.cuda.is_available() else "cpu"    

    #create a resultfolder with current date and time inside output folder
    resultfolder = "result" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputpath = os.path.join(args.output_folder, resultfolder)
    if not os.path.exists(outputpath):
            os.makedirs(outputpath)

    #create a logging file and start logging the steps
    loghandler = LogHandler(name = "Rijo", basepath=outputpath)

    #preprocess and save images
    image_preprocessing.build_model(args= args, loghandler = loghandler, resultfolder= resultfolder)

    #insance segmentation create and save  
    if args.inst_segmentation == 'cellpose':
        cellpose_segmentation.build_model(args = args, loghandler = loghandler, resultfolder=outputpath)
    elif args.inst_segmentation == 'unetwatershed':
        unet_seg_watershed.build_model(args = args, loghandler = loghandler, resultfolder=outputpath, devicetype= device_type)

    #tracking create and save
    tracking_methods.build_model(args=args, loghandler = loghandler, resultfolder= outputpath, devicetype= device_type)

    #extract single cells
    if args.is_extract_mitosis_tracks:
        single_cell_extraction.build_model(args=args, loghandler = loghandler, resultfolder= outputpath, devicetype= device_type)
        if args.is_classify_single_cell_tracks:
            single_cell_classification.build_model(args=args, loghandler = loghandler, resultfolder= outputpath, devicetype= device_type)

    #classification
    if args.is_classify_cells: 
        if args.classification_method == 'resnet18':
            imagenet_classification_methods.build_model(args=args, loghandler = loghandler, resultfolder= outputpath, devicetype= device_type)





    
