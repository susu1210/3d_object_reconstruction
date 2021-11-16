import cv2
import glob
import sys
import os
from config.segmentationParameters import SEG_METHOD


def visualize(path):
    global out
    resultIDs = []

    for file in os.listdir(path + SEG_METHOD + "_results"):
        if file.endswith(".png"):
            resultIDs.append(int(file[:-4]))

    resultIDs.sort()

    for id in resultIDs:
        filename_cad = path + 'cad/%s.jpg' % id
        cad = cv2.imread(filename_cad)
        if cad is None:
            continue
        filename_result = path + SEG_METHOD + '_results/%s.png' % id
        overlay = cv2.imread(filename_result)
        cv2.addWeighted(overlay, 0.4, cad, 0.6, 0, cad)
        out.write(cad)
        # cv2.imshow("Segment", cad)


if __name__ == "__main__":

    try:
        if sys.argv[1] == "all":
            folders = glob.glob("Data_sticked/*/")
        elif sys.argv[1] + "/" in glob.glob("Data_sticked/*/"):
            folders = [sys.argv[1] + "/"]
        else:
            exit()
    except:
        exit()


    for path in folders:
        if not os.path.exists(path + SEG_METHOD + "_results/"):
            exit()

        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(path +SEG_METHOD+"_result.avi",fourcc, 20.0, (640,480))


        visualize(path)

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

