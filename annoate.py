
'''
annotate.py
===============================================================================
Interactive annotation tool with GrabCut algorithm.

This code is modified from opencv grabcut example

*    Title: Grabcut source code
*    Author: Opencv
*    Year: 2016
*    Code version: since 2.0
*    Availability: https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

USAGE:
    python annotate.py <foldername>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground
Key 'h' - To print out instruction
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 'q' - Quit the current program
Key 'b' - Go back to the previous frame
Key 's' - To save the mask and move to annotate the next image
===============================================================================
'''
import os
import numpy as np
import cv2
import sys
from config.BackFlowParameters import ANNOTATION_INTERVAL

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def print_usage():
    print "No input image directory given\n"
    print "Correct Usage: python annotate.py <foldername> \n"
    print "e.g., annotate.py Data/Cheezit"


def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print "first draw rectangle \n"
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)


# Loading images
if len(sys.argv) == 2:
    foldername = sys.argv[1] # for drawing purposes
    # print documentation
    print __doc__

else:
    print_usage()
    exit()


path = foldername + '/'
ids = []
for file in os.listdir(path+'cad'):
    if file.endswith(".jpg"):
        ids.append(int(file[:-4]))

min_id = min(ids)
max_id = max(ids)

filename = min_id

while filename < max_id:

    img = cv2.imread(path + 'cad/%s.jpg' % filename)
    img2 = img.copy()                               # a copy of original image
    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape,np.uint8)           # output image to be shown

    # input and output windows
    cv2.namedWindow('output',16)
    cv2.namedWindow('input',16)
    cv2.setMouseCallback('input',onmouse)
    cv2.moveWindow('input',img.shape[1]+10,90)

    print " Instructions: \n"
    print " Draw a rectangle around the object using right mouse button \n"

    while(1):

        cv2.imshow('output',output)
        cv2.imshow('input',img)
        k = 0xFF & cv2.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print " mark background regions with left mouse button \n"
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print " mark foreground regions with left mouse button \n"
            value = DRAW_FG
        elif k == ord('2'): # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'): # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('h'): # print helper
            print __doc__
        elif k == ord('s'): # save image
            cv2.imwrite(path + 'annotations/%s.png' % filename, mask2)
            cv2.imwrite(path + 'mask/%s.png' % filename, mask)
            filename += ANNOTATION_INTERVAL
            print "Result saved as image \n"
            break
        elif k == ord('b'): # go back to the previous frame
            if (filename - ANNOTATION_INTERVAL) >= min_id:
                filename -= ANNOTATION_INTERVAL
                break
            else:
                print "This is the first frame! \n"
        elif k == ord('q'): # quit the current program
            exit()
        elif k == ord('r'): # reset everything
            print "resetting \n"
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape,np.uint8)           # output image to be shown
        elif k == ord('n'): # segment the image
            print """ For finer touchups, mark foreground and background after pressing keys 0-3
            and again press 'n' \n"""
            if (rect_or_mask == 0):         # grabcut with rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:         # grabcut with mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img2,img2,mask=mask2)

cv2.destroyAllWindows()
