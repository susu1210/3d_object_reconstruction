"""
tracker.py
-----------

Implementation of a tracker class that performs BackFlow tracking as described in the paper

"""
# import png
import cv2
import numpy as np
from utils.gmm import GMM
from utils.gmm import GMM
from skimage.segmentation import slic
from graph import graphcut
import glob
from config.segmentationParameters import *
from PIL import Image

class Tracker():
    def __init__(self, folder, debug = False):
        self.lk_params = dict( winSize  = (60,60),
                               maxLevel = 2,criteria = (cv2.TERM_CRITERIA_COUNT,10,1e-10))
        self.folder = folder
        self.K_fgd = K_FGD
        self.K_bgd = K_BGD
        self.interval = SEG_INTERVAL
        self.gamma = GAMMA
        self.lablefrequency = ANNOTATION_INTERVAL
        self.startframe = STARTFRAME
        self.method = SEG_METHOD
        self.visualize = debug
        self.old = None
        self.old_gray = None
        self.fg_pixels = None
        self.bg_pixels = None
        self.cnt = None
        self.mask = None
        self.bgdModel = None
        self.fgdModel = None
        self.filename = 0
        self.forward = 1
        self.max_depth = 0.85#****DEPTH_THRESH*****#
        self.depth_range = 1.0
        
    def load_groundtruth(self):
        """
        Load human scribbles and ground truth annotation
        """

        
        filename_mask = self.folder+'mask/%s.png' % self.filename
        mask = cv2.imread(filename_mask)
        self.mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        filename_anno = self.folder + 'annotations/%s.png' % self.filename
        thresh = cv2.threshold(cv2.imread(filename_anno, 0), 30, 255, cv2.THRESH_BINARY)[1]
        #v2.imread: cv2.IMREAD_GRAYSCALE--It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
        #cv2.threshold (src, thresh, maxval, type): threshold=30, set pixel<thresh to 0, pixel>thresh to maxval(255).
        return (self.mask, thresh)


    def load_images(self, filename):
        """
        Load a color image by filename
        """

        
        filename_cad = self.folder + 'cad/%s.jpg' % filename
        old = cv2.imread(filename_cad)
        depth_file = self.folder + 'depth/%s.png' % filename
        # reader = png.Reader(depth_file)
        # pngdata = reader.read()
        # depth = np.array(map(np.uint16, pngdata[2]))
        # depth = np.vstack(map(np.uint16, pngdata[2]))
        depth = Image.open(depth_file)
        depth = np.array(depth,dtype=np.uint16)

        old[depth == 0] = np.array([0,0,0],dtype = np.uint8)
        # cv2.imshow("old_gray", old)
        #cut off the background with depth> depth_thresh
        old[depth>self.max_depth/self.depth_range*65535] = np.array([0,0,0],dtype = np.uint8)
        old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("old_gray",old)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return (old, old_gray)

  
    def fgdbgd_initialization(self, thresh):
        """
         Initialize or update foreground and background gmm models
        upon the provided groundtruth mask
        """
        
        #find contours in the thresholded image
        # _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for new version opencv , there are only two ouputs
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # use only the largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # convert the contour to a mask
        cnt_mask = np.zeros(thresh.shape,dtype = np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
        #cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None,
        #                 maxLevel=None, offset=None)
        #contourIdx=-1 draw all contours; thickness indicates the width of contour line, thickness=-1 -- filling mode.


        # assign the black pixels (depth cut-off) a random number so that
        # it would not be considered foreground of background
        cnt_mask[self.old_gray<10] = 5
        if self.bgdModel is None:
            # initialize
            fg_pixels = self.old[cnt_mask==1]
            bg_pixels = self.old[cnt_mask==0]
        else:
            # update
            # bg_pixels = np.delete(self.bg_pixels, np.arange(0, self.bg_pixels.size, 4),
            #                       axis = 0)
            bg_pixels = np.delete(self.bg_pixels, np.arange(0, self.bg_pixels.shape[0], 4),
                                  axis=0)
            bg_pixels = np.concatenate((bg_pixels, self.old[cnt_mask == 0][::4]),axis = 0)
            fg_pixels = self.old[cnt_mask == 1]
        print(bg_pixels.size)
        bgdModel = GMM(self.K_bgd)
        fgdModel = GMM(self.K_fgd)    
        fgdModel.initialize_gmm(fg_pixels)
        bgdModel.initialize_gmm(bg_pixels)
        
        return (fg_pixels,bg_pixels,fgdModel, bgdModel,cnt_mask, cnt)


    def get_BB(self, cnt):
        """
        Get bounding box of the foreground object
        """
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        xo,yo,wo,ho = cv2.boundingRect(cnt)

        margin = 100
        w = wo + margin
        h = ho + margin
        x = max(0,xo-margin/2)
        y = max(0,yo-margin/2)
        rect = (y,x,h,w)   
     
        return rect
    
 
    def calcu_flow(self,segments,super_pixel,lk_params):
        """
        Propogate pixels from one frame to another using super pixel guided 
        optical flow, and return consistent flow (good_new) from backward
        and forward flow
        """
        #indexs
        #ix = np.isin(segments,super_pixel)
        ix = np.in1d(segments[:], super_pixel)
        ix=np.reshape(ix,np.shape(segments))
        sure_Backgrounnnd = np.where(ix)
        indexs = np.argwhere(ix)
        indexs[:,[0, 1]] = indexs[:,[1, 0]]
        indexs = np.array(indexs).reshape((-1,1,2)).astype(np.int32)
        indexs = indexs[::12]
        
        p0 = np.float32(indexs)
        good_new = []
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old, self.current, p0,
                                                   None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(self.current, self.old, p1,
                                                    None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            good_new = []
    
            for (i,j), good_flag in zip(p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                good_new.append((int(i), int(j)))
        except:
            pass   
        return good_new

   
    def calcu_opticalFlow(self, current_gray, interval):
        """
        Compute flow vectors between the next frame and the current frame using 
        grayscale images
        """
      
        filename_future = self.folder+'cad/%s.jpg' % (self.filename + interval)
        future = cv2.imread(filename_future)
        future_gray = cv2.cvtColor(future, cv2.COLOR_BGR2GRAY)
        depth_file = self.folder + 'depth/%s.png' % (self.filename + interval)
        # reader = png.Reader(depth_file)
        # pngdata = reader.read()
        # depth = np.array(map(np.uint16, pngdata[2]))
        # depth = np.vstack(map(np.uint16, pngdata[2]))
        depth = Image.open(depth_file)
        depth = np.array(depth, dtype=np.uint16)


        future_gray[depth == 0] = np.uint8(0)
        future_gray[depth>self.max_depth/self.depth_range*65535] = np.uint8(0)
   
        flow_vector = np.zeros_like(self.old)
        flow_vector[...,1] = 255
        flow = cv2.calcOpticalFlowFarneback(current_gray, future_gray,None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_vector[...,0] = ang*180/np.pi/2
        flow_vector[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        flow_vector = cv2.cvtColor(flow_vector,cv2.COLOR_HSV2BGR)

        return flow_vector

  
    def superpixel_flow(self):
        """
        Perform over segmentation (slic superpixel) and compute optical flow guided 
        by the super pixel
        """

        self.mask[self.old_gray<10] = 5
        index_copy = np.where(self.mask == 0)
        num_Segments = 20000
        segments = slic(self.current, n_segments = num_Segments, sigma = 5)
        super_pixel = np.unique(segments[tuple(index_copy)])
        new_pixels = self.calcu_flow(segments,super_pixel,self.lk_params)
        return (new_pixels, super_pixel, segments)

    
    def update_fgdModel(self,mask2):
        """
        Update foreground model using the segmentation result
        """

        mask2[self.current_gray < 10] = 0
        thresh = cv2.threshold(mask2, 30, 255, cv2.THRESH_BINARY)[1]

        # _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=cv2.contourArea)
    
        cnt_mask = np.zeros(self.mask.shape,dtype = np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
        cnt_mask[self.current_gray < 10] = 5
    
        fgdModel = GMM(self.K_fgd)

        # the foreground model is updated with 1/10th of the new foreground pixels
        # fg_pixels = np.delete(self.fg_pixels, np.arange(0, self.fg_pixels.size, 10),axis=0)
        fg_pixels = np.delete(self.fg_pixels, np.arange(0, self.fg_pixels.shape[0], 10), axis=0)

        fg_pixels = np.concatenate((fg_pixels, self.current[cnt_mask==1][::10]),axis=0)
    
        fgdModel.initialize_gmm(fg_pixels)
       
        return cnt, fgdModel, fg_pixels
    
  
    def update_mask(self,segments,super_pixel):
        """
        Update mask with the segmentation result and grow
        background labels
        """
        
        mask_pb=np.zeros(self.mask.shape,dtype=np.uint8)
        cv2.drawContours(mask_pb, [self.cnt], -1, 1, -1)
        
        # assign the depth cut-off pixels to some number rather
        # than 0 or 1 so that it wouldn't be considered bgd/fgd
        mask_pb[self.current_gray < 10] = np.uint8(5)
   
        index_pbbackground = np.where(mask_pb == 0)
        sp_possibles = segments[index_pbbackground]
        sp_possible = np.unique(segments[index_pbbackground])
        difference = set(super_pixel).symmetric_difference(sp_possible)
    
        if len(difference)>0:
            for sp in difference:
                num_pixels = np.where(segments==sp)
                if np.count_nonzero(sp_possibles == sp) == len(num_pixels[0]):
                    if sp+1 in super_pixel or sp-1 in super_pixel:
                        self.mask[num_pixels] = 0

        cv2.drawContours(self.mask, [self.cnt], -1, 1, -1)
        return self.mask

    
    def contour2segment(self, cnt):
        """
        Obtain segments from a single contour.
        """
        
        # draw the white overlay for the foreground
        overlay = np.zeros(self.mask.shape,dtype=np.uint8)
        cv2.drawContours(overlay, [cnt], -1, 255, -1)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        return overlay

  
    def visualization(self, overlay, background, visualize = False):
        """
        If visualize is on, show foreground mask (white overlay)  and background
        pixels using red dots.
        """
        if visualize:
            show = self.current.copy()
                        
            # draw background labels as red dots
            for center in background:
                cv2.circle(show, (int(center[0]), int(center[1])), 2,
                           (0, 0, 255), 1)
            cv2.addWeighted(overlay, 0.4, show, 0.6, 0, show)
            cv2.namedWindow('Segment',16)
            cv2.imshow("Segment", show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
            
        

    
    def run(self):
        """
        Run the tracker
        """

        num_frames = len(glob.glob1(self.folder + "cad","*.jpg")) 
    
        self.filename = self.startframe
        
        while True:

            if self.filename > num_frames - 1 - self.interval:
                print( "Progress: 100%" )
                return

            # use ground truth labels every self.lablefrequency
            elif (self.filename - self.startframe)%self.lablefrequency == 0:
                # print out progress
                if self.forward > 0:
                    print("Progress: " + str((self.filename - self.startframe)*1.0 \
                                             /(num_frames - self.startframe)*100) + "%")

                self.old, self.old_gray = self.load_images(self.filename)
                self.mask, segment = self.load_groundtruth()
                
                # Initialize foreground and background models
                s=segment.shape
                m=segment.min()
                mm=segment.max()
                self.fg_pixels, self.bg_pixels, self.fgdModel, self.bgdModel, mask_old, self.cnt  = \
                    self.fgdbgd_initialization(segment)
                    
                self.filename += self.forward*self.interval
                
            else:
                # load images
                self.current, self.current_gray = self.load_images(self.filename)
                
                # get an estimate BB for object in the next frame (so that the graph
                # cut is only applied within the BB), this is handy when the object
                # to track is small 

                rect = self.get_BB(self.cnt)
                
                # compute flow vectors from the current frame to the future frame
                # specified by the interval
                flow_vector = self.calcu_opticalFlow(self.current_gray, self.forward*self.interval)

                # propogate background labels to the new frame
                new_bgdpixels, super_pixel, segments = self.superpixel_flow()

                # update background labels for segmentation
                self.mask = np.ones(mask_old.shape,dtype = np.uint8)
                for point in new_bgdpixels:
                    try: 
                        self.mask[int(point[1])][int(point[0])] = 0
                    except:
                        pass

                mask2 = graphcut(self.current,self.mask,rect,self.fgdModel,self.bgdModel,
                                 flow_vector,self.gamma)
                
                # obtain the largest connected component as the foreground
                # update foreground models with the segmentation result
                self.cnt, self.fgdModel, self.fg_pixels = self.update_fgdModel(mask2)
                
                # update masks and grow background labels
                
                self.mask = self.update_mask(segments,super_pixel)

                # get segments and visualize results (if visualize is on)
                overlay = self.contour2segment(self.cnt)
                self.visualization(overlay, new_bgdpixels, visualize = self.visualize)

                # save results to folder
                filename = self.folder+self.method+'_results/%s.png' % self.filename
                cv2.imwrite(filename, overlay)
                

                #update globals
                self.old_gray = self.current_gray
                self.old = self.current
                if ((self.filename - self.startframe)%int(self.lablefrequency/2) == 0) and ((self.filename - self.startframe)%self.lablefrequency != 0):
                    self.filename += int((self.lablefrequency/2))
                    self.forward = - self.forward
                else:
                
                    self.filename += self.forward*self.interval

                


