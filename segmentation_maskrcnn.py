# Imports
import fnmatch
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from PIL import Image
from IPython.display import display
import cv2
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as Tf
from config.segmentationParameters import *
import random

class MaskRCNN():
    def __init__(self, path, force_cpu=True):
        self.method = SEG_METHOD
        self.path = path
        self.ycb_list = [
            "YcbCrackerBox", "YcbSugarBox", "YcbTomatoSoupCan",
            "YcbMustardBottle", "YcbGelatinBox", "YcbPottedMeatCan"]
        self.num_classes = len(self.ycb_list)+1
        self.model = self.get_instance_segmentation_model(self.num_classes)
        if torch.cuda.is_available() and (not force_cpu):
            self.model.load_state_dict(torch.load('clutter_maskrcnn_model.pt'))
            self.device = torch.device('cuda')
        else:
            self.model.load_state_dict(torch.load('clutter_maskrcnn_model.pt', map_location='cpu'))
            self.device = torch.device('cpu')
        self.model.eval()
        self.model.to(self.device)
        self.object_name = self.path.split("/")[-2]

    def get_instance_segmentation_model(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model

    def open_image(self, id):
        filename = os.path.join(self.path, "cad", str(id) + ".jpg")
        img = Image.open(filename).convert("RGB")
        return img

    def get_batch(self):
        id = 0
        batch_tensor = []
        while True:
            try:
                img = self.open_image(id)
                batch_tensor.append(Tf.to_tensor(img).to(self.device))
                id += 1
            except:
                return batch_tensor

    def run(self):
        if self.object_name in self.ycb_list:
            for i in range(len(self.ycb_list)):
                if self.ycb_list[i]==self.object_name:
                    self.target_label = i
                    break
        else:
            print("Undetectable object for trained MaskRCNN, please use BackFlow segmentation!")
            return None

        X_batch = self.get_batch()
        for i in range(STARTFRAME, len(X_batch), SEG_INTERVAL):
            with torch.no_grad():
                prediction = self.model([X_batch[i]])
            N = np.where(prediction[0]['labels'].cpu().numpy() == self.target_label)[0]
            if N.size == 0:
                N = prediction[0]['labels'].cpu().numpy()
                global pos
                pos = (0,0)
                # N = np.argmax(prediction[0]['scores'].cpu().numpy())
                img = np.asarray(self.open_image(i))

                cv2.namedWindow('image with boxes')

                boxes = prediction[0]['boxes'].cpu().numpy()
                for j in range(N.size):
                    x1, y1, x2, y2 = boxes[j]
                    rgb = (random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255))
                    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)), rgb)
                    cv2.putText(img, str(j), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
                cv2.imshow('image with boxes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                def mouse(event, x, y, flags, param):
                    global pos
                    if event == cv2.EVENT_LBUTTONDOWN:
                        xy = "%d,%d" % (x, y)
                        pos = (x,y)
                        cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=-1)
                        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (255, 255, 255), thickness=1)
                        cv2.imshow('image with boxes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                cv2.setMouseCallback('image with boxes', mouse)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                for j in range(N.size):
                    if pos[0]>=boxes[j][0] and pos[1]>=boxes[j][1] and pos[0]<=boxes[j][2] and pos[1]<=boxes[j][3]:
                        N = j
                        break
                    else:
                        N = None

            else:
                N = N[1]
                # N = prediction[0]['labels'].cpu().numpy()
                # pos = (0, 0)
                # # N = np.argmax(prediction[0]['scores'].cpu().numpy())
                # img = np.asarray(self.open_image(i))
                #
                # cv2.namedWindow('image with boxes')
                #
                # boxes = prediction[0]['boxes'].cpu().numpy()
                # for j in range(N.size):
                #     x1, y1, x2, y2 = boxes[j]
                #     rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), rgb)
                #     cv2.putText(img, str(j), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
                # cv2.imshow('image with boxes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                #
                # def mouse(event, x, y, flags, param):
                #     global pos
                #     if event == cv2.EVENT_LBUTTONDOWN:
                #         xy = "%d,%d" % (x, y)
                #         pos = (x, y)
                #         cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=-1)
                #         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                #                     1.0, (255, 255, 255), thickness=1)
                #         cv2.imshow('image with boxes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                #
                # cv2.setMouseCallback('image with boxes', mouse)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # for j in range(N.size):
                #     if pos[0] >= boxes[j][0] and pos[1] >= boxes[j][1] and pos[0] <= boxes[j][2] and pos[1] <= boxes[j][
                #         3]:
                #         N = j
                #         break
                #     else:
                #         N = None
            if N is not None:
                filename = os.path.join(self.path, self.method+"_results", str(i) + ".png")
                # mask =  np.asarray((prediction[0]['masks'][N, 0]>0.7).float().mul(255).byte().cpu().numpy(),dtype=np.uint8)
                mask = np.asarray(prediction[0]['masks'][N, 0].mul(255).byte().cpu().numpy(), dtype=np.uint8)
                mask = Image.fromarray(mask)
                mask.save(filename)
        return None

