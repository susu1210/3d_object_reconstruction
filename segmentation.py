"""
BackFlow.py
---------------

Main Function for performing video segmentatino as described in the paper

"""


import cv2
import glob
from tracker import Tracker
from segmentation_maskrcnn import MaskRCNN
import sys
import os
from config.segmentationParameters import SEG_METHOD

def print_usage():

    print( "Usage: segmentation.py <path>")
    print( "path: [data_path]/all or name of the folder")
    print( "e.g., segmentation.py Data/all, segmentation.py Data_new/all, "
           "segmentation.py Data/Cheezit")

def visualize(path):
    global out
    resultIDs = []
    
    for file in os.listdir(path +SEG_METHOD+"_results"):
        if file.endswith(".png"):
            resultIDs.append(int(file[:-4]))

    resultIDs.sort()

    for id in resultIDs:
        filename_cad = path + 'cad/%s.jpg' % id
        cad = cv2.imread(filename_cad)
        filename_result = path + SEG_METHOD+'_results/%s.png' % id
        overlay = cv2.imread(filename_result)
        cv2.addWeighted(overlay, 0.4, cad, 0.6, 0, cad)
        out.write(cad)
        cv2.imshow("Segment", cad)

    
if __name__ == "__main__":
    

    try:
        if sys.argv[1][-3:] == "all":
            folders = glob.glob(sys.argv[1][:-3]+"*/")
        elif sys.argv[1]+"/" in glob.glob("Data_new/*/") \
                or sys.argv[1]+"/" in glob.glob("Data/*/")\
                or sys.argv[1]+"/" in glob.glob("Data_stuck/*/"):
            folders = [sys.argv[1]+"/"]
        else:
            print_usage()
            exit()
    except:
        print_usage()
        exit()
    # folders=["Data/YcbGelatinBox/"]
    # folders = ["Data/cheezit/"]
    for path in folders:
        if not os.path.exists(path+SEG_METHOD+"_results/"):
            os.makedirs(path+SEG_METHOD+"_results/")

        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(path +SEG_METHOD+"_result.avi",fourcc, 20.0, (640,480))

        print(path)

        if SEG_METHOD == "BackFlow":
            tracker = Tracker(path, debug = True)

            tracker.run()
        else:
            maskrcnn = MaskRCNN(path)
            maskrcnn.run()

        visualize(path)

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print( "Results saved in " + path + SEG_METHOD+"_results")
